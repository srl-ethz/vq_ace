"""
Create the pipeline but only for exporting the model to onnx
"""

import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from srl_il.pipeline.training import TrainPipeline
import onnx
import os

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

class DecoderExportPipeline(TrainPipeline):
    """
    This pipeline works with exporting ACT decoder models, because it reads `self.algo.decoder_group_keys` and `self.algo.export_onnx` from the algo.
    """
    ## override unnecessary functions
    def _init_wandb(self, **cfg):
        pass

    ## disable mujoco visulizer
    def _init_visualizer(self, *cfg, **kwarg):
        pass

    # load the training checkpoint with training_cfg.resume_path
    # this logic is defined in TrainPipeline._init_workspace
    def run(self):
        assert(self.resume_path is not None)
        # remove the suffix of the resume_path and append "_onnx" to it
        path_elements = self.resume_path.split("/")
        assert(path_elements[-2] == "checkpoints")
        path_elements[-2] = "exported"
        os.makedirs("/".join(path_elements[:-1]), exist_ok=True)
        path_elements[-1] = ".".join(path_elements[-1].split(".")[:-1]) + ".onnx"
        out_name = "/".join(path_elements)

        self.algo.eval_epoch_begin(self.epoch)
        eval_loader_iter = iter(self.eval_loader)
        batch, mask_batch = next(eval_loader_iter)
        self.algo.export_onnx(batch, mask_batch, self.algo.decoder_group_keys, out_name)

        # load onnx and check if it is correct
        onnx_model = onnx.load(out_name)
        onnx.checker.check_model(onnx_model)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('cfg'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    pipeline = DecoderExportPipeline(**cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
