import os
import sys
from srl_il.algo.base_algo import Algo
from srl_il.pipeline.pipeline_base import Pipeline, AlgoMixin
import hydra
from omegaconf import OmegaConf
import torch

class PolicyExporter(Pipeline, AlgoMixin):
    """
    This pipeline works with exporting ACT decoder models, because it reads `self.algo.decoder_group_keys` and `self.algo.export_onnx` from the algo.
    """
    def _init_workspace(self, **cfg):
        cfg["output_dir"] = None
        super()._init_workspace(**cfg)
        

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.algo, Algo), "algo should be an instance of Algo"
        assert self.resume
        checkpoint = torch.load(self.resume_path)
        self.algo.deserialize(checkpoint['algo_state']["model"])


def get_policy_from_ckpt(ckpt_path, full = False):
    """
    Factory function to create the policy
    ckpt_path: # the path of the checkpoint, the config should be in the same directory
    example usage:
        ```
        policy = get_policy_from_ckpt("xxxx.pth")
        policy.reset_policy()
        action = policy.predict_action({
            "qpos": xxx,
            "xxximages": xxx
        })
        ```
    """
    model_dir = os.path.dirname(ckpt_path)
    # First, check in the model_dir
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_path):
        # If not found, check in the parent directory
        config_path = os.path.join(os.path.dirname(model_dir), "config.yaml")
        if not os.path.exists(config_path):
            # Raise an error if config.yaml is not found in either location
            raise FileNotFoundError(
                "config.yaml not found in model directory or its parent directory"
            )
    print("Loading config from", config_path)
    cfg = OmegaConf.load(config_path)
    cfg.resume_path=ckpt_path
    cfg.resume=True
    pipeline = PolicyExporter(**cfg)

    algo = pipeline.algo
    algo.set_eval()
    return algo
