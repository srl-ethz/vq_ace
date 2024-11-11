import torch
from ..common.autoinit_class import AutoInit

#####
# Data Augmentation Mixins
#####

class DataAgumentationBase:
    def __init__(self, **kwargs):
        pass

    def augment_train(self, data, mask):
        raise NotImplementedError
    
    def augment_eval(self, data, mask):
        return data, mask

class DataAugmentationNoiseGaussian(DataAgumentationBase):
    def __init__(self, mean=0.0, std=1.0, **kwargs):
        self.mean = mean
        self.std = std

    def augment_train(self, data, mask):
        return data + torch.randn_like(data) * self.std + self.mean, mask

class DataAugmentationTrajectorySpeed(DataAgumentationBase):
    """
    randomly slow down or speed up the trajectory
    input trajectory: [N, T, D]
    """
    def __init__(self, T_input, T_min, T_max, T_output, **kwargs):
        """
        The most naive implementation: 
        args:
            T_input: the input length of the trajectory. The input trajectory is [N, T_input, D] This information is just for assertion use.
            T_min: the minimum output length of the trajectory. When speed up the trajectory, the trajectory is interpolated to a length >= T_min
            T_max: the maximum output length of the trajectory. When slow down the trajectory, the trajectory is interpolated to a length <= T_max
            T_output: the output length of the trajectory. The output trajectory is [N, T_output, D]
            constraint: T_output <= T_min <= T_input <= T_max, 
        """
        self.T_input = T_input
        self.T_min = T_min
        self.T_max = T_max
        self.T_output = T_output
        assert T_output <= T_min <= T_input <= T_max, "T_output <= T_min <= T_input <= T_max"

    def augment_train(self, data, mask):
        data_permute = data.permute(0, 2, 1) # [N, D, T]
        mask_permute = mask.unsqueeze(1) # [N, 1, T]
        N, D, T = data_permute.shape
        assert T == self.T_input, "The input trajectory length is not correct"
        new_t = torch.randint(self.T_min, self.T_max, (N,))
        data_out = torch.zeros(N, D, self.T_output, device=data.device)
        mask_out = torch.zeros(N, 1, self.T_output, device=mask.device, dtype=mask.dtype)
        for t in range(self.T_min, self.T_max+1): # here we assume that T_max - T_min is much smaller than batch size
            selected = new_t == t
            data_out[selected] = torch.nn.functional.interpolate(data_permute[selected], size=t, mode="linear", align_corners=False)[:,:,:self.T_output]
            mask_out[selected] = torch.nn.functional.interpolate(mask_permute[selected].to(torch.uint8), size=t, mode="nearest")[:,:,:self.T_output].to(mask.dtype)
        return data_out.permute(0, 2, 1), mask_out.squeeze(1)
    
    def augment_eval(self, data, mask):
        return data[:, :self.T_output], mask[:, :self.T_output]


class DataAugmentation_6Drotation2Ang(DataAgumentationBase):
    """
    This is the action translator in diffusion policy code. 
    Note: this only work for absolute actions
    Output action: ee_pos(3) ee_rot(6) ee_gripper(1)
    Source action: ee_pos(3) ee_rot(3) ee_gripper(1)
    """
    def __init__(self,  **kwargs):
        from ..common.rotation_transformer import RotationTransformer
        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    def augment_train(self, data, mask):
        return self.augment(data, mask)

    def augment_eval(self, data, mask):
        return self.augment(data, mask)

    def augment(self, data, mask):
        actions = data
        is_dual_arm = False
        if data.shape[-1] == 14:
            # dual arm
            data = data.reshape(-1,2,7)
            is_dual_arm = True

        pos = data[...,:3]
        rot = data[...,3:6]
        gripper = data[...,6:]
        rot = self.rotation_transformer.forward(rot)
        data = torch.concatenate([
            pos, rot, gripper
        ], axis=-1)
    
        if is_dual_arm:
            data = data.reshape(-1,20)
        actions = data
        return actions, mask


class DataAugmentation:
    def __init__(self, data_augments):
        self.data_augments_cfg = data_augments
        self.augs = []
        for data_augment in data_augments:
            daug_type = data_augment["type"]
            outputname = data_augment["outname"]
            if daug_type == "gaussian_noise":
                self.augs.append(DataAugmentationNoiseGaussian(**data_augment))
            if daug_type == "trajectory_speed":
                self.augs.append(DataAugmentationTrajectorySpeed(**data_augment))
            if daug_type == "abs_action_rot_2_6d":
                self.augs.append(DataAugmentation_6Drotation2Ang(**data_augment))

    def augment_train(self, batch, mask_batch):
        for aug, data_augment in zip(self.augs, self.data_augments_cfg):
            outputname = data_augment["outname"]
            inputname = data_augment.get("inputname", outputname)
            batch[outputname], mask_batch[outputname] = aug.augment_train(batch[inputname], mask_batch[inputname])
        return batch, mask_batch

    def augment_eval(self, batch, mask_batch):
        for aug, data_augment in zip(self.augs, self.data_augments_cfg):
            outputname = data_augment["outname"]
            inputname = data_augment.get("inputname", outputname)
            batch[outputname], mask_batch[outputname] = aug.augment_eval(batch[inputname], mask_batch[inputname])
        return batch, mask_batch

class DataAugmentationMixin(AutoInit, cfgname_and_funcs=(("data_augmentation_cfg", "_init_data_augmentation"),)):
    """
    Data augmentation is to augment the batch data loaded from the dataset. 
    The augmentation is just for the purpose of training, and should not be exported when applying the model.
    """
    
    def _init_data_augmentation(self, data_augments):
        """
        Initialize the data augmentation configuration.
        args:
            data_augments: A list of dicts, each dict contains the following keys
                outname: the name of the output data what writes to the batch
                type: the type of the augmentation
                inputname: the name of the input data in batch, if none, the input is the same as the output name
                **kwargs: the parameters of the augmentation
        """
        self.data_augmentor = DataAugmentation(data_augments)
    
    def data_augmentation_train(self, batch, mask_batch):
        """
        Augment the batch data
        """
        return self.data_augmentor.augment_train(batch, mask_batch)
    
    def data_augmentation_eval(self, batch, mask_batch):
        """
        Augment the batch data
        """
        return self.data_augmentor.augment_eval(batch, mask_batch)
