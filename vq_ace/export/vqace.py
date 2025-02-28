import os
import sys
from vq_ace.algo.act import ACT_VQ
from vq_ace.pipeline.training import TrainPipeline
import hydra
from omegaconf import OmegaConf
import torch

class DecoderExporter(TrainPipeline):
    """
    This pipeline works with exporting ACT decoder models, because it reads `self.algo.decoder_group_keys` and `self.algo.export_onnx` from the algo.
    """
    ## override unnecessary functions
    def _init_wandb(self, **cfg):
        pass

    ## disable mujoco visulizer
    def _init_visualizer(self, *cfg, **kwarg):
        pass

    ## override the dataset initialization
    def _init_dataset(self, **kwargs):
        pass


def factory(model_path, full = False):
    """
    Factory function to create the decoder
    model_path: # the path of the checkpoint, the config should be in the same directory

    """
    model_dir = os.path.dirname(model_path)
    config_path=os.path.join(model_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.resume_path=model_path
    cfg.resume=True
    pipeline = DecoderExporter(**cfg)

    algo = pipeline.algo
    algo.set_eval()
    algo.set_device("cuda")
    if not full:
        def model(z, qpos):
            """
            z: bs, 1 
            qpos: bs, 11
            """
            with torch.no_grad():
                z = torch.concat([z, torch.zeros((z.shape[0], 4), dtype=z.dtype, device=z.device)], axis=1)
                z = algo._models['vq']._embedding(z)
                batch={
                    "qpos": qpos.unsqueeze(1),
                }
                batch_mask = {
                    "qpos": torch.ones_like(batch["qpos"][:,:,0], dtype=torch.bool),
                }
                out = algo.decode(z, batch, batch_mask)
                return out[:,:10]
        return model
    else:
        def model(z, qpos):
            """
            z: bs, 5 
            qpos: bs, 11
            """
            with torch.no_grad():
                z = algo._models['vq']._embedding(z)
                batch={
                    "qpos": qpos.unsqueeze(1),
                }
                batch_mask = {
                    "qpos": torch.ones_like(batch["qpos"][:,:,0], dtype=torch.bool),
                }
                out = algo.decode(z, batch, batch_mask)
                return out[:,:]
        return model

def algo_factory(model_path):
    model_dir = os.path.dirname(model_path)
    config_path=os.path.join(model_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.resume_path=model_path
    cfg.resume=True
    pipeline = DecoderExporter(**cfg)

    algo = pipeline.algo
    algo.set_eval()
    algo.set_device("cuda")
    return algo, cfg

if __name__ == "__main__":
    model_path = "/home/chenyu/workspace/vqbet_ws/media/vqace_model_20240904-123427/checkpoint_100.pth"
    model = factory(model_path)
    z = torch.randint(0, 4, (10, 1)).cuda()
    qpos = torch.randn(10, 11).cuda()
    out = model(z, qpos)
    print(out)