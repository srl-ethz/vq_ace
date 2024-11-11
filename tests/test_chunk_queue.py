import unittest
import torch
from srl_il.common.chunk_buffer import ChunkBuffer, TemporalAggregationBuffer, ChunkBufferBatch

class TestChunkBuffer(unittest.TestCase):
    def test_append_and_get_top(self):
        # Define dimensions
        B, T, H, W, C = 3, 0, 4, 5, 2
        chunk_dim = 1
        chunk_length = 2
        max_length = 6

        # Initialize the shape (Note: T is set to max_length for buffer initialization)
        shape = (B, -1, H, W, C)

        # Initialize the ChunkBuffer
        queue = ChunkBuffer(shape, chunk_dim, chunk_length, max_length)

        # Create chunks with known values
        chunk_shape = list(shape)
        chunk_shape[chunk_dim] = 1  # Each chunk has length 1 along chunk_dim

        chunk1 = torch.rand(chunk_shape) * 1
        chunk2 = torch.rand(chunk_shape) * 2
        chunk3 = torch.rand(chunk_shape) * 3

        # Append chunks
        queue.append(chunk1)
        data, mask = queue.get_top()
        self.assertTrue(torch.equal(data[:,[1],...], chunk1), "Data does not match expected data")
        expected_mask = torch.cat([torch.zeros(chunk1.shape[:chunk_dim+1], dtype=torch.bool), torch.ones(chunk1.shape[:chunk_dim+1])], dim=chunk_dim)
        self.assertTrue(torch.equal(mask, expected_mask), "Mask does not match expected mask")

        queue.append(chunk2)
        queue.append(chunk3)

        # Get top and validate
        data, mask = queue.get_top()

        # Expected data is chunk2 and chunk3 concatenated along chunk_dim
        expected_data = torch.cat([chunk2, chunk3], dim=chunk_dim)

        # Assert data correctness
        self.assertTrue(torch.equal(data, expected_data), "Data does not match expected data")

        # Expected mask is all ones before rollback
        expected_mask = torch.ones(data.shape[:queue._chunk_dim+1], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected_mask), "Mask does not match expected mask")

        # Append more chunks to trigger rollback
        chunk4 = torch.rand(chunk_shape)
        chunk5 = torch.rand(chunk_shape)
        chunk6 = torch.rand(chunk_shape)
        chunk7 = torch.rand(chunk_shape)

        queue.append(chunk4)
        queue.append(chunk5)
        queue.append(chunk6)
        queue.append(chunk7)

        # Get top after rollback
        data, mask = queue.get_top()

        # Expected data is chunk6 and chunk7 concatenated along chunk_dim
        expected_data = torch.cat([chunk6, chunk7], dim=chunk_dim)

        # Assert data correctness after rollback
        self.assertTrue(torch.equal(data, expected_data), "Data after rollback does not match expected data")

        # Expected mask is all ones after rollback
        expected_mask = torch.ones(data.shape[:queue._chunk_dim+1], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected_mask), "Mask after rollback does not match expected mask")


        chunk8 = torch.rand(chunk_shape)
        queue.append(chunk8)
        data, mask = queue.get_top()
        expected_data = torch.cat([chunk7, chunk8], dim=chunk_dim)
        self.assertTrue(torch.equal(data, expected_data), "Data after append does not match expected data")

class TestChunkBufferBatch(unittest.TestCase):
    def test_reset_idx(self):
        # Parameters
        batch_size = 2
        data_shape = (3,6)
        chunk_length = 2
        max_length = 5
        device = torch.device('cpu')
        # Initialize the buffer
        buffer = ChunkBufferBatch(
            batch_size=batch_size,
            data_shape=data_shape,
            chunk_length=chunk_length,
            max_length=max_length,
            device=device
        )
        data0 = torch.rand((batch_size, *data_shape),  dtype=torch.float32, device=device)
        data1 = torch.rand((batch_size, *data_shape),  dtype=torch.float32, device=device)
        data2 = torch.rand((batch_size, *data_shape),  dtype=torch.float32, device=device)
        buffer.append(data0)
        data, mask = buffer.get_top()
        self.assertTrue(torch.equal(data[:, 1, :], data0), "Data does not match expected data")
        self.assertTrue(mask[:, 1].all(), "Mask does not match expected mask")
        self.assertTrue((~mask[:, 0]).all(), "Mask does not match expected mask")
        buffer.append(data1)
        data, mask = buffer.get_top()
        ground_truth_data = torch.stack([data0, data1], dim=1)
        self.assertTrue(torch.equal(data, ground_truth_data), "Data does not match expected data")
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)), "Mask does not match expected mask")
        buffer.reset_idx(0)
        buffer.append(data2)
        data, mask = buffer.get_top()
        self.assertTrue(mask[1,:].all(), "Mask does not match expected mask") # all mask for batch 1 should be True
        self.assertTrue(mask[0,1].all(), "Mask does not match expected mask") # the new mask for batch 0 should be True
        self.assertTrue((~mask[0,0]).all(), "Mask does not match expected mask") # the old mask for batch 0 is False



class TestTemporalAggregationBuffer(unittest.TestCase):
    def test_append_and_get_top(self):
        # Parameters
        batch_size = 2
        data_shape = (1,)  # Action dimension
        chunk_length = 2
        max_timesteps = 5
        device = torch.device('cpu')

        # Initialize the buffer
        buffer = TemporalAggregationBuffer(
            batch_size=batch_size,
            data_shape=data_shape,
            chunk_length=chunk_length,
            max_timesteps=max_timesteps,
            device=device
        )

        total_timesteps = 10  # Number of timesteps to test (including beyond max_timesteps)

        for t in range(total_timesteps):
            # Create data with known values for testing
            calcuated_time = t
            data = torch.full(
                (batch_size, chunk_length, *data_shape),
                calcuated_time,
                device=device
            ) + torch.arange(t, t + chunk_length).reshape(1, chunk_length, 1) * 0.1
            # each data value is {calcuated_time . execution_time}
            
            
            # Append data
            buffer.append(data)
            retrieved_data, retrieved_mask = buffer.get_top()

            # Compute expected data and mask
            # Expected data: the data from buffer at positions [t - chunk_length : t] along time dimension
            expected_data_list = []
            expected_mask_list = []
            for idx in range(max(t - chunk_length+1, 0), t+1):
                calcuated_time = idx  # Since data_value = t + 1
                expected_data_list.append(torch.full((batch_size, *data_shape), calcuated_time, dtype=torch.float32, device=device)) 
                expected_mask_list.append(torch.ones((batch_size,), dtype=torch.bool, device=device))
            
            expected_data = torch.stack(expected_data_list, dim=1)
            expected_data += t * 0.1
            expected_mask = torch.stack(expected_mask_list, dim=1)

            # print(buffer.buffer[0].squeeze(2))
            # print("retrieved_data", retrieved_data)
            # print("expected_data", expected_data)
            # Compare retrieved data and expected data
            self.assertTrue(
                torch.allclose(retrieved_data, expected_data),
                f"Retrieved data at time {t} does not match expected data"
            )

            # Compare retrieved mask and expected mask
            self.assertTrue(
                torch.equal(retrieved_mask, expected_mask),
                f"Retrieved mask at time {t} does not match expected mask"
            )

        print("Test append and get_top passed.")

    def test_reset_idx(self):
        # Parameters
        batch_size = 2
        data_shape = (3,6)
        chunk_length = 2
        max_timesteps = 5
        device = torch.device('cpu')
        # Initialize the buffer
        buffer = TemporalAggregationBuffer(
            batch_size=batch_size,
            data_shape=data_shape,
            chunk_length=chunk_length,
            max_timesteps=max_timesteps,
            device=device
        )
        data0 = torch.rand((batch_size, chunk_length, *data_shape),  dtype=torch.float32, device=device)
        data1 = torch.rand((batch_size, chunk_length, *data_shape),  dtype=torch.float32, device=device)
        data2 = torch.rand((batch_size, chunk_length, *data_shape),  dtype=torch.float32, device=device)
        buffer.append(data0)
        buffer.append(data1)
        data, mask = buffer.get_top()
        ground_truth_data = torch.stack([data0[:,1,:,:], data1[:,0,:,:]], dim=1)
        self.assertTrue(torch.equal(data, ground_truth_data), "Data does not match expected data")
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)), "Mask does not match expected mask")
        buffer.reset_idx(0)
        buffer.append(data2)
        data, mask = buffer.get_top()
        self.assertTrue(mask[1,:].all(), "Mask does not match expected mask") # all mask for batch 1 should be True
        self.assertTrue(mask[0,1].all(), "Mask does not match expected mask") # the new mask for batch 0 should be True
        self.assertTrue((~mask[0,0]).all(), "Mask does not match expected mask") # the old mask for batch 0 is False

if __name__ == '__main__':
    unittest.main()
