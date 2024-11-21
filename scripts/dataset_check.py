"""
Create the dataset and dataloaders for training and testing
Calculate the statistics of the dataset and act as a sanity check
use:
python scripts/check_dataset.py --config-name=xxxxxxx
"""

import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from srl_il.pipeline.pipeline_base import Pipeline, DatasetMixin, AlgoMixin, NormalizationMixin
from srl_il.pipeline.data_augmentation import DataAugmentationMixin
from srl_il.algo.base_algo import Algo
import os
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


class DataCheckPipeline(Pipeline, DatasetMixin, AlgoMixin, NormalizationMixin, DataAugmentationMixin):

    def _init_algo(self, **algo_cfg):
        # Hack the init_algo function from AlgoMixin
        self.algo = Algo(algo_cfg={"device": "cpu"})
        self.num_sample_train = 2
        self.num_sample_eval = 2

    def show_seq_lowdim(self, data, mask, name_prefix):
        """
        data: (B, T, D)
        Plot each dimension with matplotlib
        """
        B,T,D = data.shape
        for bi in range(B):
            for di in range(D):
                plt.plot(data[bi, :, di], label=f"dim{di}")
            plt.legend()
            plt.savefig(f"{name_prefix}_{bi}.jpg")
            plt.close()

    def show_seq_image(self, data, mask, name_prefix):
        """
        data: (B, T, C, H, W)
        Save the images as mp4
        """
        B,T,C,H,W = data.shape
        data = (data*255.0).to(torch.uint8)
        small_h = 128
        small_w = (W * small_h) // H
        for bi in range(B):
            output_path = f"{name_prefix}_{bi}.gif"
            img_frames = [Image.fromarray(data[bi, ti].permute(1,2,0).cpu().numpy()).resize(size=(small_w, small_h)) for ti in range(T)]
            img_frames[0].save(output_path, format='gif', save_all=True, append_images=img_frames[1:], optimize=True, duration=100, loop=0)


    def run(self):
        train_iter = iter(self.train_loader)
        for ind in range(self.num_sample_train):
            batch, mask_batch = next(train_iter)
            batch, mask_batch = self.data_augmentation_train(batch, mask_batch)
            for key, value in batch.items():
                if value.dim() == 3 and key in mask_batch.keys():
                    self.show_seq_lowdim(value, mask_batch[key],
                        os.path.join(self.output_dir, f"train_{ind}_{key}"))
                if value.dim() == 5 and key in mask_batch.keys():
                    self.show_seq_image(value, mask_batch[key],
                        os.path.join(self.output_dir, f"train_{ind}_{key}"))

        eval_iter = iter(self.eval_loader)
        for ind in range(self.num_sample_eval):
            batch, mask_batch = next(eval_iter)
            batch, mask_batch = self.data_augmentation_eval(batch, mask_batch)
            for key, value in batch.items():
                if value.dim() == 3 and key in mask_batch.keys():
                    self.show_seq_lowdim(value, mask_batch[key],
                        os.path.join(self.output_dir, f"eval_{ind}_{key}"))
                if value.dim() == 5 and key in mask_batch.keys():
                    self.show_seq_image(value, mask_batch[key],
                        os.path.join(self.output_dir, f"eval_{ind}_{key}"))

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('srl_il', 'cfg'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cfg.dataset_cfg.batch_size = 2
    # alter the keys_traj so that the dataloader loads the whole window
    for i in range(len(cfg.dataset_cfg.data.keys_traj)):
        cfg.dataset_cfg.data.keys_traj[i][2] = None # start from beginning
        cfg.dataset_cfg.data.keys_traj[i][3] = None # end at the end
    pipeline = DataCheckPipeline(**cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
