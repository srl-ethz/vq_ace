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
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def dataloader_statistics(dataloader):
    """
    Calculate the statistics of the dataloader
    """
    sum_batches = [None, ]* 2
    sum_batches_squared = [None, ]* 2
    cnt_batches = 0
    for batch in dataloader:
        for i, dataname in enumerate(['traj', 'mask']):
            if sum_batches[i] is None:
                sum_batches[i] = {k:b.sum(0).sum(0) for k,b in batch[i].items()}
                sum_batches_squared[i] = {k: (b**2).sum(0).sum(0) for k,b in batch[i].items()}
            else:
                for k in batch[i].keys():
                    sum_batches[i][k] += batch[i][k].sum(0).sum(0)
                    sum_batches_squared[i][k] += (batch[i][k] ** 2).sum(0).sum(0)
        cnt_batches += list(batch[0].values())[0].shape[0]
    
    for i, dataname in enumerate(['traj', 'mask']):
        print(f"Statistics for {dataname}")
        for k in sum_batches[i].keys():
            mean = sum_batches[i][k] / cnt_batches
            std = torch.sqrt(sum_batches_squared[i][k] / cnt_batches - mean**2)
            print(f"{k}'s element, shape: {mean.shape}, total count: {cnt_batches}")
            print(f"Mean: {mean}")
            print(f"Std: {std}")



class DataCheckPipeline(Pipeline, DatasetMixin, AlgoMixin, NormalizationMixin, DataAugmentationMixin):

    def _init_algo(self, **algo_cfg):
        # Hack the init_algo function from AlgoMixin
        self.algo = Algo(algo_cfg={"device": "cpu"})


    def run(self):
        print("Statistics for train loader")
        dataloader_statistics(self.train_loader)
        print("Statistics for test loader")
        dataloader_statistics(self.test_loader)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('cfg'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    pipeline = DataCheckPipeline(**cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
