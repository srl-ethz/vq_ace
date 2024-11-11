import unittest
import torch
import os
import tempfile

import torch.utils
from srl_il.algo.base_algo import Algo, TrainerMixin
from srl_il.pipeline.training import TrainPipeline
from srl_il.models.common.linear_normalizer import LinearNormalizer
from torch.utils.data import TensorDataset
from omegaconf import OmegaConf

class DummyAlgo(Algo):
    def _init_algo(self, device):
        super()._init_algo(device)
        self._models['nn'] = torch.nn.Linear(10, 5)


class DummyTrainer(DummyAlgo, TrainerMixin):
    def eval_epoch_begin(self, epoch):
        pass
    
    def eval_epoch_end(self, epoch):
        pass

    def eval_step(self, inputs, epoch=None):
        pass

    def train_epoch_begin(self, epoch):
        pass

    def train_epoch_end(self, epoch):
        pass

    def train_step(self, inputs, epoch=None):
        pass
    

class DummyDatasets:
    def __init__(self):
        self.train_data = TensorDataset(torch.randn(100, 10))
        self.val_data = TensorDataset(torch.randn(100, 10))
        self.test_data = TensorDataset(torch.randn(100, 10))


# test the serialize function and deserialize function inherited from Algo
class TestSaveAndLoad(unittest.TestCase):
    pipelinecfg = dict(
        seed=42,
        output_dir=None,
        algo_cfg=dict(
            _target_ = f"{__name__}.DummyTrainer",
            algo_cfg = dict(
                device="cpu",
            ),
            trainer_cfg = dict(
                optimizer_cfg=dict(
                    nn=dict(
                        optm_cls="torch.optim.Adam",
                        lr = 0.001
                    )
                )
            )
        ),
        dataset_cfg=dict(
            data = dict(
                _target_ = f"{__name__}.DummyDatasets"
            ),
            batch_size = 10,
            pin_memory = False,
            num_workers = 0
        ),
        lr_scheduler_cfg=dict(
            nn=dict(
                type="torch",
                scheduler_cls="torch.optim.lr_scheduler.ExponentialLR",
                params=dict(
                    gamma=0.9
                )
            )
        ),
        wandb_cfg=dict(
            project="dummy",
            entity="dummy",
            run_name="dummy",
            mode="disabled"
        ),
        visualizer_cfg=dict(visualizer = None, visualizer_key = None),
        normalizer_cfg=dict(
            foo = {
                "type": "hardcode",
                "mean": 1.233,
                "std": 0.567
            },
        ),
        training_cfg=None,
        data_augmentation_cfg=dict(
            data_augments=[]
        )
    )

    def test_save_and_load_checkpoint(self):
        cfg = OmegaConf.create(self.pipelinecfg)
        pipeline = TrainPipeline(
            **cfg
        )
        
        # make changes to the pipeline:
        pipeline.algo._models['nn'].weight.data.fill_(0.5)
        pipeline.algo._normalizers['foo'].params_dict["offset"] = torch.tensor(0.528)
        pipeline.algo._optimizers['nn'].step()
        pipeline._lr_schedulers["nn"].step()
        pipeline.epoch = 10
        pipeline.best_train_loss = 0.5
        
        # Use a temporary file for saving and loading
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            filepath = tmp_file.name

            # Save the model
            pipeline.save_checkpoint(filepath)

            # Create a new instance to load the model
            resume_cfg = self.pipelinecfg.copy()
            resume_cfg['resume_path'] = filepath
            resume_cfg["normalizer_cfg"]["foo"]["mean"] = 0.0
            resume_cfg["normalizer_cfg"]["foo"]["std"] = 1.0
            resume_cfg = OmegaConf.create(resume_cfg)
            new_pipeline = TrainPipeline(
                **resume_cfg
            )

        # Assert
        self.assertEqual(new_pipeline.epoch, 10)
        self.assertEqual(new_pipeline.best_train_loss, 0.5)
        self.assertTrue(torch.allclose(new_pipeline.algo._models['nn'].weight.data, torch.tensor(0.5)))
        self.assertEqual(new_pipeline.algo._normalizers['foo'].params_dict["offset"], 0.528)
        self.assertEqual(new_pipeline.algo._normalizers['foo'].params_dict["scale"], 0.567)
        self.assertAlmostEqual(new_pipeline.algo._optimizers['nn'].param_groups[0]['lr'], 0.0009)
        self.assertEqual(new_pipeline._lr_schedulers["nn"].last_epoch, 1)


    def test_save_and_load_trainer(self):
        cfg = OmegaConf.create(self.pipelinecfg)

        algo = DummyTrainer(**cfg.algo_cfg)
        loss = algo._models['nn'](torch.randn(10, 10)).sum()
        loss.backward()
        algo._normalizers['foo'] = LinearNormalizer(torch.tensor(0.234), torch.tensor(3.371))
        algo._optimizers['nn'].step()

        algo_weight = algo._models['nn'].weight.data.clone()
        algo_optim_exp_avg = algo._optimizers['nn'].state[algo._models['nn'].weight]['exp_avg'].clone()

        states = algo._get_model_and_optimizer_states()
        newcfg = cfg.algo_cfg.copy()
        newcfg['trainer_cfg']['optimizer_cfg']['nn']['lr'] = 999
        new_algo = DummyTrainer(**cfg.algo_cfg)
        new_algo._load_model_and_optimizer_states(states)
        
        self.assertTrue(torch.allclose(new_algo._models['nn'].weight.data, algo_weight))
        self.assertTrue(torch.allclose(new_algo._optimizers['nn'].state[new_algo._models['nn'].weight]['exp_avg'], algo_optim_exp_avg))
        self.assertEqual(new_algo._normalizers['foo'].params_dict["offset"], 0.234)
        self.assertEqual(new_algo._normalizers['foo'].params_dict["scale"], 3.371)
        self.assertEqual(new_algo._optimizers['nn'].param_groups[0]['lr'], 0.001)
    

    def test_save_and_load_model(self):
        cfg = OmegaConf.create(self.pipelinecfg)
        algo = DummyTrainer(**cfg.algo_cfg)
        loss = algo._models['nn'](torch.randn(10, 10)).sum()
        loss.backward()
        algo._normalizers['foo'] = LinearNormalizer(torch.tensor(0.4872), torch.tensor(3.574))
        algo._optimizers['nn'].step()

        algo_weight = algo._models['nn'].weight.data.clone()

        states = algo.serialize()
        newcfg = cfg.algo_cfg.copy()
        
        new_algo = DummyTrainer(**cfg.algo_cfg)
        new_algo.deserialize(states)
        
        self.assertTrue(torch.allclose(new_algo._models['nn'].weight.data, algo_weight))
        self.assertEqual(new_algo._normalizers['foo'].params_dict["offset"], 0.4872)
        self.assertEqual(new_algo._normalizers['foo'].params_dict["scale"], 3.574)
        

if __name__ == '__main__':
    unittest.main()
