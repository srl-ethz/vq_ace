import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod
from ...common.autoinit_class import AutoInit
import hydra
import torch 
from PIL import Image
import wandb

class MujocoDataWriter(ABC):
    def _init_mujocovis(self, mujoco_vis):
        pass

    @abstractmethod
    def write_mj_data(self, data_mujoco, joints):
        raise NotImplementedError()

class MujocoWriter6dofP2(MujocoDataWriter):

    def write_mj_data(self, data_mujoco, joints):
        # parse joints
        rot_matrix = np.array([joints[-6:-3], joints[-3:], np.cross(joints[-6:-3], joints[-3:])])
        quat = R.from_matrix(rot_matrix).as_quat()
        wrist_point = joints[16:19]
        joints = joints[:16]

        # write data
        data_mujoco.mocap_pos[0,:] = wrist_point    # IF THERE IS THE 6DOF BASE HAND
        data_mujoco.mocap_quat[0,:] = quat
        # Thumb
        data_mujoco.qpos[0] = joints[0]/2
        data_mujoco.qpos[1] = joints[0]/2
        data_mujoco.qpos[2] = joints[1]/2
        data_mujoco.qpos[3] = joints[1]/2
        data_mujoco.qpos[4] = joints[2]/2
        data_mujoco.qpos[5] = joints[2]/2
        data_mujoco.qpos[6] = joints[3]/2
        data_mujoco.qpos[7] = joints[3]/2
        # Fingers
        running_idx = 8
        for i in range(4,16):
            if i%3 == 0:
                data_mujoco.qpos[running_idx] = joints[i]/2 
                running_idx += 1
                data_mujoco.qpos[running_idx] = joints[i]/2 
                running_idx += 1
                data_mujoco.qpos[running_idx] = joints[i]/2 * 0.71
                running_idx += 1
                data_mujoco.qpos[running_idx] = joints[i]/2 * 0.71
                running_idx += 1
            else:
                data_mujoco.qpos[running_idx] = joints[i]/2 
                running_idx += 1
                data_mujoco.qpos[running_idx] = joints[i]/2 
                running_idx += 1

class MujocoWriterQposAccordingToNames(MujocoDataWriter):
    def __init__(self, joint_names):
        self.joint_names = joint_names
    
    def _init_mujocovis(self, mujoco_vis):
        self.model = mujoco_vis.model
        self.ids = [self.model.joint(j).id for j in self.joint_names]

    def write_mj_data(self, data_mujoco, joints):
        # write data
        for i, id in enumerate(self.ids):
            data_mujoco.qpos[id] = joints[i]


class MujocoWriterStablePosCtrlAccordingToNames(MujocoDataWriter):
    def __init__(self, actuator_names):
        self.actuator_names = actuator_names
    
    def _get_actuator_id(self, name):
        try:
            return self.model.actuator(name).id
        except:
            print(f"Warning: actuator {name} not found in the mujoco model")
            return None

    def _init_mujocovis(self, mujoco_vis):
        self.model = mujoco_vis.model
        self.ids = [self._get_actuator_id(name) for name in self.actuator_names]
            

    def write_mj_data(self, data_mujoco, ctrl):
        # write data
        converged = False
        assert(len(ctrl) == len(self.ids), "The length of ctrl should be the same as the length of actuator_names")
        mujoco_ctrl = data_mujoco.ctrl.copy()
        for i, id in enumerate(self.ids):
            if id is not None:
                mujoco_ctrl[id] = ctrl[i]
        counter = 0
        while not converged and counter < 100:
            counter += 1
            q_pos_last = data_mujoco.qpos.copy()
            data_mujoco.ctrl = mujoco_ctrl.copy()
            mujoco.mj_step(self.model, data_mujoco)
            converged = np.allclose(q_pos_last, data_mujoco.qpos, atol=1e-6)
        
class MujocoVisualizer:
    def __init__(self, model_path, cam_cfg, data_writer):
        self.cam_cfg = cam_cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.data_writer = data_writer
        self.data_writer._init_mujocovis(self)

        self.window_size = self.cam_cfg.get("window_size", [256, 256])
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = self.cam_cfg.get("lookat", [0, 0, 0])
        self.cam.distance = self.cam_cfg.get("distance", 1)
        self.cam.azimuth = self.cam_cfg.get("azimuth", 0)
        self.cam.elevation = self.cam_cfg.get("elevation", -45)
        
        self.renderer = mujoco.Renderer(self.model, height=self.window_size[1], width=self.window_size[0])

    def generate_gif(self, joint_sequence):
        model = self.model
        data = self.data
        renderer = self.renderer

        image_list = []
        for joint in joint_sequence:
            self.data_writer.write_mj_data(data, joint)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, self.cam)
            image = renderer.render()
            image_list.append(image)
        return image_list

def sample_gauss(mu, log_var):
    std = torch.exp(0.5 * log_var)
    # Sample N(0, I)
    eps = torch.randn_like(std)
    return mu + eps * std

def image_list_to_wandb(image_list, fps=10):
    import cv2
    import tempfile
    height, width, layers = image_list[0].shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'VP80') # Codec for mp4
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_video:
        temp_video_path = temp_video.name
        video = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        for frame in image_list:
            video.write(frame)
        video.release()
    return wandb.Video(temp_video_path)

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def gif_contact(gif_dict):
    """
    gif_dict: {key: image_list}
    """

    # Get the keys (subtitles) and image lists from the dictionary
    keys = list(gif_dict.keys())
    images_list = list(gif_dict.values())

    # Determine the number of frames in the GIF
    num_frames = len(images_list[0])

    # Determine the shape of the individual images
    img_height, img_width, _ = images_list[0][0].shape

    # Prepare the font for subtitles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 1

    # Calculate the height of the subtitle area
    subtitle_height = 30

    # List to hold the concatenated frames
    concatenated_frames = []

    # Loop through each frame
    for i in range(num_frames):
        # Concatenate images horizontally
        concatenated_image = np.hstack([images_list[j][i] for j in range(len(keys))])

        # Convert the image to PIL for adding subtitles
        pil_img = Image.fromarray(concatenated_image)
        draw = ImageDraw.Draw(pil_img)

        # Calculate the width of the concatenated image
        concatenated_width = concatenated_image.shape[1]

        # Add subtitles below each image
        x_offset = 0
        for key, images in gif_dict.items():
            text_size = cv2.getTextSize(key, font, font_scale, font_thickness)[0]
            text_x = x_offset + (img_width - text_size[0]) // 2
            text_y = subtitle_height + text_size[1] // 2

            # Add text to the image
            draw.text((text_x, text_y), key, fill=(255, 255, 20))

            x_offset += img_width

        # Convert back to NumPy array
        concatenated_frame_with_text = np.array(pil_img)

        # Append the frame with subtitles
        concatenated_frames.append(concatenated_frame_with_text)

    return concatenated_frames


class MujocoVisualizerMixin(AutoInit, cfgname_and_funcs=(("visualizer_cfg", "_init_visualizer"),)):
    def _init_visualizer(self, visualizer, visualizer_key):
        """
        Visulize trajectories using mujoco.
        Set Attributes:
            self.mj_visualizer
        """
        self.mj_visualizer = hydra.utils.instantiate(visualizer)
        self.mj_visualizer_key = visualizer_key

    def draw_gif_from_joint_sequence(self, joint_sequence_dict,
                                    output_path = None, wandb_log_name = None):
        """
        Draw gif from joint sequence.
        args:
            joint_sequence_dict: {key: joint_sequence}. The joint sequences gifs will be concatenated horizontally for comparison.
        """
        imgs = {
            key: self.mj_visualizer.generate_gif(joint_sequence) for key, joint_sequence in joint_sequence_dict.items()
        }
        concatenated_frames = gif_contact(imgs)
        if output_path is not None:
            img_frames = [Image.fromarray(f) for f in concatenated_frames]
            img_frames[0].save(output_path, format='GIF', save_all=True, append_images=img_frames[1:], optimize=False, duration=100, loop=0)
        if wandb_log_name is not None:
            wandb.log({wandb_log_name: wandb.Video(np.transpose(np.array(concatenated_frames), (0, 3, 1, 2)), fps=10, format="gif")})
