import unittest
from srl_il.dataset.dataset_base import TrajectoryDataset, SequenceDataset, get_train_val_test_seq_datasets
import torch

class DumbTrajDataset(TrajectoryDataset):
    def __init__(self, trajs_lengths):
        self.trajs_lengths = trajs_lengths
        self.traj_key1 = [torch.arange(l).float() for l in trajs_lengths] # shape (traj_length, )
        self.traj_key2 = [torch.arange(l).float()[:,None] + torch.tensor([[1, 100]]) for l in trajs_lengths] # shape (traj_length, 2)
        self.global_key1 = [torch.tensor([0,1,2,3])+i for i,_ in enumerate(trajs_lengths)]

    def __getitem__(self, idx):
        return (
            {"tkey1": self.traj_key1[idx], "tkey2": self.traj_key2[idx]}, # traj dict
            {"gkey1": self.global_key1[idx]} # global dict
        )

    def __len__(self):
        return len(self.trajs_lengths)
    
    def get_seq_length(self, idx):
        return self.trajs_lengths[idx]

    def load(self, data, key, is_global=False):
        return data

class TestModule1(unittest.TestCase):

    def test_no_padding(self):
        trajs_lengths = [3, 5, 7]
        dataset = DumbTrajDataset(trajs_lengths)
        seq_dataset = SequenceDataset(dataset, window_size=4, 
                keys_traj=[("tkey2", "tkey2", None, None), ("tkey1", "tkey1", 1, 3)], keys_global=["gkey1"], 
                pad_before=False, pad_after=False,
                pad_type="zero")
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(seq_dataset), 6) # 0 + 2 + 4
        for i in range(len(seq_dataset)):
            traj_dict, traj_masks = seq_dataset[i]
            self.assertEqual(traj_dict["tkey1"].shape, (2, ))
            self.assertEqual(traj_dict["tkey2"].shape, (4, 2))
            self.assertEqual(traj_masks["tkey1"].shape, (2,))
            self.assertEqual(traj_masks["tkey2"].shape, (4,))
            self.assertEqual(traj_dict["gkey1"].shape, (4,))

        # test the value
        traj_dict, traj_masks = seq_dataset[1] # this is the second idx in the traj of length 5
        torch.testing.assert_close(traj_dict["tkey1"], torch.tensor([2,3]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([1,2,3,4]).float()[:,None] + torch.tensor([[1, 100]]) )
        torch.testing.assert_close(traj_masks["tkey1"], torch.tensor([True,True]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([True, True,True, True]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([1,2,3,4]))


    def test_both_padding(self):
        trajs_lengths = [3, 5]
        dataset = DumbTrajDataset(trajs_lengths)
        seq_dataset = SequenceDataset(dataset, window_size=4, 
                keys_traj=[("tkeyout1", "tkey1", 1, 3), ("tkey2", "tkey2", None, None)], keys_global=["gkey1"], 
                pad_before=True, pad_after=True,
                pad_type="zero")
        self.assertEqual(len(seq_dataset), 14) # 6 + 8
        for i in range(len(seq_dataset)):
            traj_dict, traj_masks = seq_dataset[i]
            self.assertEqual(traj_dict["tkeyout1"].shape, (2, ))
            self.assertEqual(traj_dict["tkey2"].shape, (4, 2))
            self.assertEqual(traj_masks["tkeyout1"].shape, (2,))
            self.assertEqual(traj_masks["tkey2"].shape, (4,))
            self.assertEqual(traj_dict["gkey1"].shape, (4,))

        # test the value
        traj_dict, traj_masks = seq_dataset[0] # this is the first idx, should have 3 zeros in the beginning
        torch.testing.assert_close(traj_dict["tkeyout1"], torch.tensor([ 0,0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[0, 0], [0, 0], [0, 0], [1, 100]]).float())
        torch.testing.assert_close(traj_masks["tkeyout1"], torch.tensor([ False, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([False, False, False, True]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[1] # this is the second idx, should have 2 padded zeros in the beginning
        torch.testing.assert_close(traj_dict["tkeyout1"], torch.tensor([ 0, 0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[0, 0], [0, 0], [1, 100], [2, 101]]).float())
        torch.testing.assert_close(traj_masks["tkeyout1"], torch.tensor([ False, True]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([False, False, True, True]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[4] # this is should have 2 padded zeros in the end
        torch.testing.assert_close(traj_dict["tkeyout1"], torch.tensor([ 2, 0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[2, 101], [3, 102], [0, 0], [0, 0]]).float())
        torch.testing.assert_close(traj_masks["tkeyout1"], torch.tensor([ True, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([True, True, False, False]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[5] # this is should have 3 padded zeros in the end
        torch.testing.assert_close(traj_dict["tkeyout1"], torch.tensor([ 0, 0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[3, 102], [0, 0], [0, 0], [0, 0]]).float())
        torch.testing.assert_close(traj_masks["tkeyout1"], torch.tensor([ False, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([True, False, False, False]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[6] # this is should be the first idx of the second traj
        torch.testing.assert_close(traj_dict["tkeyout1"], torch.tensor([ 0,0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[0, 0], [0, 0], [0, 0], [1, 100]]).float())
        torch.testing.assert_close(traj_masks["tkeyout1"], torch.tensor([ False, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([False, False, False, True]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([1,2,3,4]))


    def test_near_padding(self):
        # same as test both padding, but use near padding
        trajs_lengths = [3, 5]
        dataset = DumbTrajDataset(trajs_lengths)
        seq_dataset = SequenceDataset(dataset, window_size=4, 
                keys_traj=[("tkey1", "tkey1", 1, 3), ("tkey2", "tkey2", None, None)], keys_global=["gkey1"], 
                pad_before=True, pad_after=True,
                pad_type="near")

        # test the value
        traj_dict, traj_masks = seq_dataset[0] # this is the first idx, should have 3 zeros in the beginning
        torch.testing.assert_close(traj_dict["tkey1"], torch.tensor([ 0, 0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[1, 100], [1, 100], [1, 100], [1, 100]]).float())
        torch.testing.assert_close(traj_masks["tkey1"], torch.tensor([ False, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([False, False, False, True]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[1] # this is the second idx, should have 2 padded zeros in the beginning
        torch.testing.assert_close(traj_dict["tkey1"], torch.tensor([ 0, 0 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[1, 100], [1, 100], [1, 100], [2, 101]]).float())
        torch.testing.assert_close(traj_masks["tkey1"], torch.tensor([ False, True]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([False, False, True, True]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[4] # this is should have 2 padded zeros in the end
        torch.testing.assert_close(traj_dict["tkey1"], torch.tensor([ 2, 2 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[2, 101], [3, 102], [3, 102], [3, 102]]).float())
        torch.testing.assert_close(traj_masks["tkey1"], torch.tensor([ True, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([True, True, False, False]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))

        traj_dict, traj_masks = seq_dataset[5] # this is should have 3 padded zeros in the end
        torch.testing.assert_close(traj_dict["tkey1"], torch.tensor([ 2, 2 ]).float())
        torch.testing.assert_close(traj_dict["tkey2"], torch.tensor([[3, 102], [3, 102], [3, 102], [3, 102]]).float())
        torch.testing.assert_close(traj_masks["tkey1"], torch.tensor([ False, False]))
        torch.testing.assert_close(traj_masks["tkey2"], torch.tensor([True, False, False, False]))
        torch.testing.assert_close(traj_dict["gkey1"], torch.tensor([0,1,2,3]))


    def test_train_val_test_split(self):
        trajdataset = DumbTrajDataset([10, 10])
        train, val, test = get_train_val_test_seq_datasets(trajdataset, 
                                                           0.1, 0.2, # test_fraction, val_fraction
                                                           4, 10, # window_size_train, window_size_test,
                                                           keys_traj=[("tkey1", "tkey1", None, None)],
                                                           keys_global=["gkey1"],
                                                           pad_before=True, 
                                                           pad_after=True,
                                                           pad_type="zero"
                                                           )
        train_sample, val_sample, test_sample = train[0], val[0], test[0]
        torch.testing.assert_close(train_sample[0]["gkey1"],  val_sample[0]["gkey1"]) # train and val have the same global value, because they are from the same trajectory
        torch.testing.assert_close(train_sample[0]["gkey1"] + test_sample[0]["gkey1"], torch.tensor([1,3,5,7])) # train and test are from different trajectory 

    def test_dataloaders(self):
        trajdataset = DumbTrajDataset([10, 10, 10, 10])
        train, val, test = get_train_val_test_seq_datasets(trajdataset, 
                                                           0.1, 0.2, # test_fraction, val_fraction
                                                           4, 4, # window_size_train, window_size_test,
                                                           keys_traj=[("tkey1", "tkey1", 1, 3), ("tkey2", "tkey2", None, None)], keys_global=["gkey1"], 
                                                           pad_before=True, 
                                                           pad_after=True,
                                                           pad_type="zero"
                                                           )

        # test the dataloaders
        train_loader = torch.utils.data.DataLoader(train, batch_size=3, shuffle=True)#, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val, batch_size=3, shuffle=True)#, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test, batch_size=3, shuffle=True)#, collate_fn=collate_fn)
        for loader in [train_loader, val_loader, test_loader]:
            batch_traj_tuple, batch_traj_masks = next(iter(loader))
            self.assertEqual(batch_traj_tuple["tkey1"].shape, (3, 2, ))
            self.assertEqual(batch_traj_tuple["tkey2"].shape, (3, 4, 2))
            self.assertEqual(batch_traj_masks["tkey1"].shape, (3, 2))
            self.assertEqual(batch_traj_masks["tkey2"].shape, (3, 4))
            self.assertEqual(batch_traj_tuple["gkey1"].shape, (3, 4))

if __name__ == '__main__':
    unittest.main()