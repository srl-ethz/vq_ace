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

