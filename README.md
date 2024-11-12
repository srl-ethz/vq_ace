# srl_il
This is a code base designed for rapid prototyping and benchmarking of imitation learning techniques. With its flat configuration system, the users can easily test out their new data, configuration, algorithm, and ideas.

## Get started

This section walks you through installation and running robomimic tasks, the following is tested on Ubuntu22.04

```bash
conda create -n srl_il python=3.11 # or other versions of python
conda activate srl_il

pip install git+https://github.com/ARISE-Initiative/robomimic.git@9273f9c # install the latest robomimic by the time this is document is writen. Note, this is different v0.3.0

cd srl_il
pip install -e .[robomimic] # will install optional robomimic dependencies
```

### Download the dataset 
Find [here](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html#method-1-using-download-datasets-py-recommended) the instructions for downloading dataset


### Train and rollout

```bash
python3 scripts/run_pipeline.py --config-name=imitation_learning_robomimic_act  preset_robomimic=lift_image  dataset_cfg.data.data_path=my_data_path/lift/ph/image.hdf5
```

The command will runs the act algorithm, with the task `lift_image`, the path of the dataset is specified in `dataset_cfg.data.data_path`

Similarly, one can also try out diffusion policy with

```bash
python3 scripts/run_pipeline.py --config-name=imitation_learning_robomimic_diffusion  preset_robomimic=lift_image  dataset_cfg.data.data_path=my_data_path/lift/ph/image.hdf5
```

### Optional: try position control
Robomimic only provides datasets with relative control commands. Diffusion policy main page provides the data with absolute positional command ([download page](https://diffusion-policy.cs.columbia.edu/data/training/))

Make the following changes to the configuration files to allow position control.

1. Replace the `data.data_path` of config files in `preset_robomimic` to the absolute dataset from diffusion policy, called `image_abs.hdf5`
2. Change the `target_dims` of the config file to 10. This is because diffusion policy uses 6d representation for rotation.
3. Change the `env.env_kwargs.control_delta` to `false`
4. Change the `algo_cfg.policy_cfg.policy_translator` to the following: It convert the 6d rotation from the policy to 3d
    - `_target_: srl_il.algo.base_algo.PolicyTranslator_6Drotation2Ang`
    - `action_name: actions`
5. Add the following to `data_augmentation_cfg.data_augments`: It convert the 3d rotation from dataloader to 6d
    - `- outname: actions # this is used for absolute action`
    - `type: abs_action_rot_2_6d`
6. Change the `normalizer_cfg.actions`. So that the normalizer is computed based on the statistics after data augmentation (i.e. after converting from 3d to 6d)
    - `type: augmentor_stats`
    - `min_max: true` 
    - `max_data_points: 100000` 

## Use your dataset

### Implement the trajectory dataset

Create a trajectory dataset that inherits from `srl_il.dataset.dataset_base.TrajectoryDataset`. 
You just need to implement your trajectory dataset that iterates over each trajectory (they can be of variable length). 
The `SequenceDataset` will do the slicing and padding for you.

In `TrajectoryDataset`, each item is a trajectory, it should have the following interfaces implemented
- `__getitem__(idx)`: returns a tuple that contains two dicts (traj_dict, global_dict)
  - traj_dict: **key**: dataname, **value**: data of the shape (seq_len, data_dim1, data_dim2, ...). Note that the values of the same dict should have the same seq_len.
  - global_dict: **key**: dataname, **value**: data of the shape (data_dim1, data_dim2, ...)
- `get_seq_length(idx)`: returns the seq_len of `__getitem__(idx)`
- `__len__`: the number of trajectories contained in this dataset
- `load(data, key, is_global)`: This function is used to turn the data returned from `__getitem__` and sliced by sequence dataset into torch Tensors in memory. 

> We have this `load` function for the sake of performance. Suppose the data is saved as `hdf5` in disk, then `h5_file['data'][a:b]` is only a reference to the data, and does not load all the data from the disk to memory, until it is convert to numpy or torch.
> So the best practice without loading all dataset to memory is: return the reference of the full sequence in `__getitem__`, slice the sequence in Sequence dataset, and load the sliced data into memory with `load` function.
> If you are interested, please take a look at the `RobomimicTrajectorySequenceDatasetPreloaded` and `RobomimicTrajectorySequenceDataset` implementation in `dataset.robomimic_dataset.py`.


### Configure the dataloading

The function `get_train_val_test_seq_datasets` expect `window_size`, `keys_traj` and `keys_global` to specify the length of each slice, and which slices of data to load from the dataset. 
This part can be best explained with examples
```yaml
  window_size_train: 21
  keys_traj: [['img0', 'agentview_image', 1, 2],  # use the second frame as the observation
              ['img1', 'robot0_eye_in_hand_image', 1, 2],  
              ['robot0_eef_pos', 'robot0_eef_pos', null, 2],  
              ['actions', 'actions', 1, null] # use the rest of the frames as the action
            ] 
```
In this example, the trajectory will be cutted into slices with length 21. However, not all steps in these slices are need for a traning. So these two indexes in `kyes_traj` specifies which portion of the slice is loaded. In the example, the dataloader loads the second `agentview_image` and `robot0_eye_in_hand_image` in the window, loads the first `robot0_eef_pos` of the first two timesteps, and load the action sequence starting from the second one.

So the dataloader returns dict with the following keys and shapes.
- `img0`: `(B, 1, C, H, W)`
- `img1`: `(B, 1, C, H, W)`
- `robot0_eef_pos`: `(B, 2, 3)`
- `actions`: `(B, 20, 7)`

This is equivalent to the following setting in diffusion policy, but it offers more possibilities in data loading
```yaml
horizon: 21
n_obs_steps: 2 # in the encoder drop the first images
n_action_steps: 20
n_latency_steps: 0
```


## 🙏Acknowledgement
- The robomimic tasks and observation encoders are adapted from [Robomimic](https://github.com/ARISE-Initiative/robomimic)
- The linear normalizer implementation is adapted from [diffusion policy](https://github.com/real-stanford/diffusion_policy)
- The vector quantize implementation is adapted from [vq_bet_officia](https://github.com/jayLEE0301/vq_bet_official)

