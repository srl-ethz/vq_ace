import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from ...common.autoinit_class import AutoInit
from PIL import Image
import os
from scipy.spatial.transform import Rotation 

def project_points(points, extrinsics, intrinsics):
    """
    Project 3D points into 2D using extrinsics and intrinsics.

    Args:
        points (np.ndarray): Array of shape (N, 3) or (3,) for a single point.
        extrinsics (np.ndarray): 4x4 extrinsics matrix.
        intrinsics (np.ndarray): 3x3 camera intrinsics matrix.

    Returns:
        np.ndarray: Projected 2D points, shape (N, 2) or (2,).
    """
    rvec = cv2.Rodrigues(np.array(extrinsics[:3, :3]))[0]
    tvec = np.array(extrinsics[:3, 3])
    projected, _ = cv2.projectPoints(np.array(points), rvec, tvec, np.array(intrinsics), np.zeros(5))
    return projected.squeeze()


def draw_trajectory(images, trajectory, color=(0, 255, 0), thickness=2):
    """
    Draw the trajectory as a line on the image.

    Args:
        images (list of np.ndarray): The input images. Each image is the image to draw at certain time step.
            To visualize the whole trajectory into one image, pass the same image multiple times.
        trajectory (np.ndarray): Array of shape (N, 2) or (2,) for a single point.
        color (tuple): Color of the trajectory line.
        thickness (int): Thickness of the trajectory line.
    """
    for image, point in zip(images, trajectory):
        pt = tuple(map(int, point))
        cv2.circle(image, pt, radius=3, color=color, thickness=thickness)


def draw_orientation(images, points, directions, extrinsics, intrinsics, scale=0.07, color=(255, 0, 0)):
    """
    Draw orientation arrows at specific points.

    Args:
        images (list of np.ndarray): The input images. Each image is the image to draw at certain time step.
            To visualize the whole trajectory into one image, pass the same image multiple times.
        points (np.ndarray): Array of shape (N, 3) or (3,) for a single point.
        directions (np.ndarray): Array of shape (N, 3) or (3,) for a single orientation.
        extrinsics (np.ndarray): 4x4 extrinsics matrix.
        intrinsics (np.ndarray): 3x3 camera intrinsics matrix.
        scale (float): Scale factor for arrow length.
        color (tuple): Color of the arrows.
    """

    projected_points = project_points(points, extrinsics, intrinsics)
    for i, (image, point, orientation) in enumerate(zip(images, projected_points, directions)):
        # Calculate the end point for the arrow
        end_point_3d = points[i] + scale * orientation
        end_point_2d = project_points(end_point_3d[np.newaxis, :], extrinsics, intrinsics)
        pt1 = tuple(map(int, point))
        pt2 = tuple(map(int, end_point_2d))
        cv2.arrowedLine(image, pt1, pt2, color, 2, tipLength=0.2)


def visualize_actions(images, extrinsics, intrinsics, gt_positions, gt_directions, pred_positions, pred_directions, chunk_size=5):
    """
    Visualize the ground truth (GT) and predicted trajectories and directions on the image. 
    This operation is done in-place.

    Args:
        images (list of np.ndarray): The input images. Each image is the image to draw at certain time step.
            To visualize the whole trajectory into one image, pass the same image multiple times.
        projection_matrix (np.ndarray): 3x4 projection matrix.
        gt_positions (np.ndarray): Array of shape (N, 3) representing GT 3D positions.
        gt_directions (np.ndarray): Array of shape (N, 3) representing GT 3D directions.
        pred_positions (np.ndarray): Array of shape (N, 3) representing predicted 3D positions.
        pred_directions (np.ndarray): Array of shape (N, 3) representing predicted 3D directions.
        chunk_size (int): Length of action chunks for visualization.
    """
    # Project and draw ground truth trajectory and directions (green)
    gt_projected = project_points(gt_positions, extrinsics, intrinsics)
    draw_trajectory(images, gt_projected, color=(0, 1.0, 0))  # Green
    for i in range(0, len(gt_positions), chunk_size):
        chunk_positions = gt_positions[i:i + chunk_size]
        chunk_directions = gt_directions[i:i + chunk_size]
        draw_orientation(images, chunk_positions, chunk_directions, extrinsics, intrinsics, color=(0, 1.0, 0))  # Green

    # Project and draw predicted trajectory and directions (red)
    pred_projected = project_points(pred_positions, extrinsics, intrinsics)
    draw_trajectory(images, pred_projected, color=(1.0, 0, 0))  # Red
    for i in range(0, len(pred_positions), chunk_size):
        chunk_positions = pred_positions[i:i + chunk_size]
        chunk_directions = pred_directions[i:i + chunk_size]
        draw_orientation(images, chunk_positions, chunk_directions, extrinsics, intrinsics, color=(1.0, 0, 0))  # Red


class ActionProjectorMixin(AutoInit, cfgname_and_funcs=(("projection_visualizer_cfg", "_init_projection_visualizer"),)):

    def _init_projection_visualizer(self, pose_key=None, img_views=None):
        """
        Initialize the projection visualizer with the specified pose keys and image views.
        args:
            pose_key: Name of the key that represent the pose in the world frame. x, y, z, qw, qx, qy, qz.
            img_views: list of image views to visualize. tuple(img_name, img_key, extrinsics_key, intrinsics_key)

        """
        self.projection_visualize_pose_key = pose_key
        self.projection_visualize_img_views = img_views
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)

    
    def visualize(self, batch, predicted_actions, epoch):
        # everything to cpu
        # predicted_actions = {k: v.detach().cpu() for k, v in predicted_actions.items()}
        # batch = {k: v.detach().cpu() for k, v in batch.items()}
        
        if self.projection_visualize_pose_key is None or self.projection_visualize_img_views is None:
            return None
        
        poses_gt = batch[self.projection_visualize_pose_key].cpu()
        poses_pred = predicted_actions[self.projection_visualize_pose_key].detach().cpu()
        B,T = poses_gt.shape[:2]
        for img_name, img_key, extrinsics_key, intrinsics_key in self.projection_visualize_img_views:
            extrinsics = batch[extrinsics_key]
            intrinsics = batch[intrinsics_key]
            images = batch[img_key]
            for i in range(B):
                extrinsics_i = extrinsics[i].cpu().numpy().reshape(4, 4)
                intrinsics_i = intrinsics[i].cpu().numpy().reshape(3, 3)
                poses_gt_i = poses_gt[i].numpy()
                poses_pred_i = poses_pred[i].numpy()
                image_i = images[i][0].cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
                image_i_gif = [image_i.copy() for _ in range(T)]
                pos_gt_i = poses_gt_i[:, :3]
                pos_pred_i = poses_pred_i[:, :3]
                direction_gt_i = Rotation.from_quat(poses_gt_i[:, 3:]).as_matrix()[:,:,2] # (x, y, z, w)
                direction_pred_i = Rotation.from_quat(poses_pred_i[:, 3:]).as_matrix()[:, :,2] # (x, y, z, w)
                visualize_actions(image_i_gif, extrinsics_i, intrinsics_i, pos_gt_i, direction_gt_i, pos_pred_i, direction_pred_i, T)
                # write gif
                img_frames = [Image.fromarray((f*255).astype(np.uint8)) for f in image_i_gif]
                output_path = os.path.join(self.output_dir, "visualizations", f"epoch_{epoch}_{img_name}_{i}.gif")
                img_frames[0].save(output_path, format='GIF', save_all=True, append_images=img_frames[1:], optimize=False, duration=100, loop=0)
        return {}

        