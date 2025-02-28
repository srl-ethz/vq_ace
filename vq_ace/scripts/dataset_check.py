"""
Create the dataset and dataloaders for training and testing
Calculate the statistics of the dataset and act as a sanity check
use:
python vq_ace/scripts/dataset_statistics.py --config-name=train_embed_faive_vae
"""

import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from vq_ace.pipeline.pipeline_base import Pipeline, DatasetMixin, AlgoMixin, NormalizationMixin
from vq_ace.pipeline.data_augmentation import DataAugmentationMixin
from vq_ace.pipeline.mujoco_visulizer import MujocoVisualizerMixin 
from vq_ace.algo.base_algo import Algo
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



class DataCheckPipeline(Pipeline, DatasetMixin, MujocoVisualizerMixin, AlgoMixin, NormalizationMixin, DataAugmentationMixin):

    def _init_algo(self, **algo_cfg):
        # Hack the init_algo function from AlgoMixin
        self.algo = Algo(algo_cfg={"device": "cpu"})


    def run(self):
        print("Statistics for train loader")
        dataloader_statistics(self.train_loader)
        print("Statistics for test loader")
        dataloader_statistics(self.test_loader)

        # self.sample_and_recons(output_dir="/home/chenyu/tmp", action_name_in_batch='action')
        batch_traj, batch_mask = next(iter(self.test_loader))
        batch_traj = {k: v[:10] for k,v in batch_traj.items()}
        batch_mask = {k: v[:10] for k,v in batch_mask.items()}
        vis_list = []
        for i, joint_sequence in enumerate(batch_traj["action"]):
            vis_list.append({"origins": joint_sequence})

        for i, vis_dict in tqdm(enumerate(vis_list)):
            self.draw_gif_from_joint_sequence(vis_dict, 
                output_path=os.path.join("/home/chenyu/tmp", f"{i}.gif")
            )

        # test the data augmentation
        batch_traj, batch_mask = next(iter(self.test_loader))
        batch_traj = {k: v[:5] for k,v in batch_traj.items()}
        batch_mask = {k: v[:5] for k,v in batch_mask.items()}
        vis_list = []
        batch_traj_eval, batch_mask_eval = self.data_augmentation_eval(deepcopy(batch_traj), deepcopy(batch_mask))
        for i, joint_sequence in enumerate(batch_traj_eval["action"]):
            vis_list.append({"eval": joint_sequence})

        for j in range(3):
            batch_traj_train, batch_mask_train = self.data_augmentation_train(deepcopy(batch_traj), deepcopy(batch_mask))
            for i, joint_sequence in enumerate(batch_traj_train["action"]):
                vis_list[i][f"train{j}"] =  joint_sequence
        for i, vis_dict in tqdm(enumerate(vis_list)):
            self.draw_gif_from_joint_sequence(vis_dict, 
                output_path=os.path.join("/home/chenyu/tmp", f"augment_check_{i}.gif")
            )


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
