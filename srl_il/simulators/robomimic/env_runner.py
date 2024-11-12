# from robomimic.envs.env_base import EnvBase
import robomimic.utils.env_utils as EnvUtils
from omegaconf import OmegaConf

from robomimic.utils.obs_utils import initialize_obs_modality_mapping_from_dict
import torch

# A simple vectorize wrapper for robomimic env. The input and output contains batch dimension = 1
class RobomimicEnv:
    def __init__(self, env_name, env_type, env_kwargs, render_video=True, obs_name_mapping=None):
        """
        Args:
            env_name (str): name of the environment
            env_type (str): type of the environment
            env_kwargs (dict): dictionary of environment kwargs
            render_video (bool): whether to render video
            obs_name_mapping (list[Tuples]): list of tuples of (obs_name_to_policy, obs_name_from_env)
        """

        initialize_obs_modality_mapping_from_dict({ # A very naive solution to use the env wrapper from robomimic
                "rgb": [
                    "agentview_image",
                    "robot0_eye_in_hand_image",
                    "robot1_eye_in_hand_image",
                    "sideview_image",
                    "shouldercamera0_image",
                    "shouldercamera1_image"
                ],
                "low_dim": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos"
                ]
            }
        )
        env_meta = {
            "env_name": env_name,
            "type": env_type,
            "env_kwargs": OmegaConf.to_container(env_kwargs),
        }

        self._env =  EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=None, # none is default which is env_meta["env_name"]
            render=False,
            render_offscreen=render_video,
            use_image_obs=env_kwargs.get("use_camera_obs", False),
            use_depth_obs=env_kwargs.get("camera_depths", False)
        )

        self._obs_name_mapping = obs_name_mapping if obs_name_mapping is not None else []

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
    
    def reset(self):
        obs_dict = self._env.reset()
        return self._process_obs_dict(obs_dict)
    
    def is_success(self):
        success_metrics = self._env.is_success()
        success_metrics = {k: torch.tensor(v)[None,...].float() for k, v in success_metrics.items()}
        return success_metrics
    
    def step(self, action):
        action = action[0].cpu().numpy()
        obs_dict, reward, done, info = self._env.step(action)
        obs_dict = self._process_obs_dict(obs_dict)
        reward = torch.tensor([reward])[None,...]
        done = torch.tensor([done])[None,...]
        info = {k: torch.tensor(v[None,...]).float() for k, v in info.items()}
        return obs_dict, reward, done, info

    def _process_obs_dict(self, obs_dict):
        obs_dict.update({
            obs_name_to_policy: obs_dict[obs_name_from_env]
            for obs_name_to_policy, obs_name_from_env in self._obs_name_mapping
        })
        obs_dict = {k: torch.tensor(v[None,...]).float() for k, v in obs_dict.items()}
        return obs_dict

    @property
    def name(self):
        return self._env.name
    
    @property
    def rollout_exceptions(self):
        return self._env.rollout_exceptions