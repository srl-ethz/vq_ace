# test algo.act
import unittest
from srl_il.algo.act import ACT, ACT_VQ
from srl_il.models.common.linear_normalizer import LinearNormalizer
import torch
from tempfile import NamedTemporaryFile
import onnx
import onnxruntime as ort
import numpy as np
import warnings

class TestACT(unittest.TestCase):

    a_dim = 2
    z_dim = 1
    T_target = 5
    T_z = 2


    # causal mode
    act_cfg = dict(
        algo_cfg = dict(
            device = "cpu",
            target_dims = {"a": a_dim},
            z_dim = z_dim,
            T_target = T_target,
            T_z = T_z,
            encoder_is_causal = True,
            decoder_is_causal = True,
            encoder_group_keys = ["obs"],
            decoder_group_keys = ["obs"],
            encoder_cfg = dict(d_model = 32),
            decoder_cfg = dict(d_model = 32)
        ),
        obs_encoder_cfg = dict(
            output_dim= 32,
            group_emb_cfg = None,
            obs_groups_cfg = dict(
                obs= dict(
                    datakeys = ["obs"],
                    encoder_cfg = {
                        "type" : "torch",
                        "_target_":  "torch.nn.Linear",
                        "in_features": 6,
                        "out_features": 32,
                    },
                    posemb_cfg = {
                        "type": "seq",
                        "seq_len": 2
                    },
                )
            )
        )
    )

    def test_causal(self):
        """
        Test the causal and no_causal mode of ACT
        """
        algo = ACT(**self.act_cfg)
        algo._normalizers['a'] = LinearNormalizer(torch.tensor(0.), torch.tensor(1.))
        algo._normalizers['obs'] = LinearNormalizer(torch.tensor(0.), torch.tensor(1.))

        algo.set_eval() # otherwise the dropout will be on and the result will be different

        self.assertEqual(algo.target_tidx, [0,1,2,3,4])
        self.assertEqual(algo.z_tidx, [2, 4])

        obs_batch = dict(
            obs = torch.randn(3, 1, 6),
        )
        mask_batch = dict(
            obs = torch.ones(3, 1)
        )
        target = torch.randn(3, 5, 2)
        obs_batch["a"] = target
        act_encoder_out= algo.encode(obs_batch, mask_batch)
        mu0 = act_encoder_out["mu"]
        
        # modify the target[0], target[1], target[2]. The corresponding mu and logvar should change
        target[0, 0] += 1.
        target[1, 2] += 1.
        target[2, 3] += 1.
        obs_batch["a"] = target
        act_encoder_out = algo.encode(obs_batch, mask_batch)
        mu1 = act_encoder_out["mu"]
        self.assertTrue(all([mu0[0,0,i] != mu1[0,0,i] for i in range(self.z_dim)]))
        self.assertTrue(all([mu0[0,1,i] != mu1[0,1,i] for i in range(self.z_dim)]))

        self.assertTrue(all([mu0[1,0,i] != mu1[1,0,i] for i in range(self.z_dim)]))
        self.assertTrue(all([mu0[1,1,i] != mu1[1,1,i] for i in range(self.z_dim)]))

        self.assertTrue(all([mu0[2,0,i] == mu1[2,0,i] for i in range(self.z_dim)])) # only batch 2 is not affected because its k_tidx is 2, less than 3
        self.assertTrue(all([mu0[2,1,i] != mu1[2,1,i] for i in range(self.z_dim)]))

        # test decode   
        z = torch.randn(3, 2, 1)
        recon0 = algo.decode(z, obs_batch, mask_batch)["a"]

        z[1, 0] += 1.
        z[2, 1] += 1.
        recon1 = algo.decode(z, obs_batch, mask_batch)["a"]

        self.assertTrue(all([recon0[0,0,i] == recon1[0,0,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[0,1,i] == recon1[0,1,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[0,2,i] == recon1[0,2,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[0,3,i] == recon1[0,3,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[0,4,i] == recon1[0,4,i] for i in range(self.a_dim)]))

        self.assertTrue(all([recon0[1,0,i] == recon1[1,0,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[1,1,i] == recon1[1,1,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[1,2,i] != recon1[1,2,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[1,3,i] != recon1[1,3,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[1,4,i] != recon1[1,4,i] for i in range(self.a_dim)]))

        self.assertTrue(all([recon0[2,0,i] == recon1[2,0,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[2,1,i] == recon1[2,1,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[2,2,i] == recon1[2,2,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[2,3,i] == recon1[2,3,i] for i in range(self.a_dim)]))
        self.assertTrue(all([recon0[2,4,i] != recon1[2,4,i] for i in range(self.a_dim)]))

    def test_onnx(self):
        act = ACT(**self.act_cfg)
        act._normalizers['a'] = LinearNormalizer(torch.randn(self.a_dim), torch.rand(self.a_dim)+1)
        act._normalizers['obs'] = LinearNormalizer(torch.randn(6), torch.rand(6)+1)

        act.set_device("cpu")
        act.set_eval() 

        # Create a NamedTemporaryFile for the ONNX file
        with NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
            bs = 3
            # Export the model to ONNX
            obs_batch = dict(
                obs = torch.randn(bs, 2, 6),
            )
            mask_batch = dict(
                obs = torch.ones(bs, 2),
            )            
            
            act.export_onnx(obs_batch, mask_batch, act.decoder_group_keys, temp_file.name)
            act.set_eval()
            bs = 256
            obs_batch = dict(
                obs = torch.randn(bs, 2, 6),
            )
            mask_batch = dict(
                obs = torch.ones(bs, 2),
            )

            z = torch.rand((bs, self.T_z, self.z_dim), device="cpu") 
            # act.set_eval() 
            with torch.no_grad():
                out = act.decode(z, obs_batch, mask_batch)["a"]

            # Load the ONNX model
            onnx_model = onnx.load(temp_file.name)

            # Check if the ONNX model is valid
            onnx.checker.check_model(onnx_model)

            # # Run inference with the ONNX model
            ort_session = ort.InferenceSession(temp_file.name)

            ort_inputs = {"z": z.numpy(),
                          "obs": obs_batch["obs"].numpy(),
                          "obs_mask": mask_batch["obs"].numpy()
                          }

            ort_outs = ort_session.run(None, ort_inputs)
            # Compare the outputs

            np.testing.assert_allclose(out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
            print("act onnx out is equal")

        

class TestACT_VQ(unittest.TestCase):

    a_dim = 2
    z_dim = 1
    T_target = 5
    T_z = 2
    num_embeddings = 6
    # causal mode
    actvq_cfg = dict(
        algo_cfg = dict(
            device = "cpu",
            target_dims = {"a": a_dim},
            z_dim = z_dim,
            T_target = T_target,
            T_z = T_z,
            encoder_is_causal = True,
            decoder_is_causal = True,
            encoder_group_keys = ["obs1"],
            decoder_group_keys = ["obs1", "obs2"],
            encoder_cfg = dict(d_model = 32),
            decoder_cfg = dict(d_model = 32),
            vq_cfg = dict(
                num_embeddings= num_embeddings,
                decay= 0.99
            )
        ),
        obs_encoder_cfg = dict(
            output_dim= 32,
            group_emb_cfg = None,
            obs_groups_cfg = dict(
                obs1= dict(
                    datakeys = ["obs1"],
                    encoder_cfg = {
                        "type" : "torch",
                        "_target_":  "torch.nn.Linear",
                        "in_features": 6,
                        "out_features": 32,
                    },
                    posemb_cfg = {
                        "type": "seq",
                        "seq_len": 2
                    },
                ),
                obs2= dict(
                    datakeys = ["obs2"],
                    encoder_cfg = {
                        "type" : "torch",
                        "_target_":  "torch.nn.Linear",
                        "in_features": 9,
                        "out_features": 32,
                    },
                    posemb_cfg = {
                        "type": "seq",
                        "seq_len": 3
                    },
                )
            )
        )
    )


    def test_onnx(self):
        act_vq = ACT_VQ(**self.actvq_cfg)
        act_vq._normalizers['a'] = LinearNormalizer(torch.randn(self.a_dim), torch.rand(self.a_dim)+1)
        act_vq._normalizers['obs1'] = LinearNormalizer(torch.randn(6), torch.rand(6)+1)
        act_vq._normalizers['obs2'] = LinearNormalizer(torch.randn(9), torch.rand(9)+1)

        act_vq.set_device("cpu")
        act_vq.set_eval() 

        # Create a NamedTemporaryFile for the ONNX file
        with NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
            bs = 3
            # Export the model to ONNX
            obs_batch = dict(
                obs1 = torch.randn(bs, 2, 6),
                obs2 = torch.randn(bs, 3, 9),
            )
            mask_batch = dict(
                obs1 = torch.ones(bs, 2),
                obs2 = torch.ones(bs, 3)
            )            
            
            act_vq.export_onnx(obs_batch, mask_batch, act_vq.decoder_group_keys, temp_file.name)
            act_vq.set_eval()
            bs = 256
            obs_batch = dict(
                obs1 = torch.randn(bs, 2, 6),
                obs2 = torch.randn(bs, 3, 9),
            )
            mask_batch = dict(
                obs1 = torch.ones(bs, 2),
                obs2 = torch.ones(bs, 3)
            )

            latent_inds = torch.randint(0, self.num_embeddings, (bs, self.T_z), device="cpu") 
            # act_vq.set_eval() 
            with torch.no_grad():
                z = act_vq._models["vq"]._embedding(latent_inds)
                z = z.view(-1, self.T_z, self.z_dim)
                out = act_vq.decode(z, obs_batch, mask_batch)["a"]

            # Load the ONNX model
            onnx_model = onnx.load(temp_file.name)

            # Check if the ONNX model is valid
            onnx.checker.check_model(onnx_model)

            # # Run inference with the ONNX model
            ort_session = ort.InferenceSession(temp_file.name)

            ort_inputs = {"embed_inds": latent_inds.numpy(),
                          "obs1": obs_batch["obs1"].numpy(),
                          "obs2": obs_batch["obs2"].numpy(),
                          "obs1_mask": mask_batch["obs1"].numpy(),
                          "obs2_mask": mask_batch["obs2"].numpy()
                          }

            ort_outs = ort_session.run(None, ort_inputs)
            # Compare the outputs

            np.testing.assert_allclose(out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
            print("out is equal")


            # Both models should raise exceptions if the latent_inds is out of range
            with self.assertRaises(Exception) as context1:
                latent_inds[0, 0] = self.num_embeddings # out of range
                z = act_vq._models["vq"]._embedding(latent_inds)
                z = z.view(-1, self.T_z, self.z_dim)
                out = act_vq.decode(z, obs_batch, mask_batch)
            
            with self.assertRaises(Exception) as context2:
                latent_inds[0, 0] = -1
                z = act_vq._models["vq"]._embedding(latent_inds)
                z = z.view(-1, self.T_z, self.z_dim)
                out = act_vq.decode(z, obs_batch, mask_batch)
            
            with self.assertRaises(Exception) as context3:
                ort_inputs["embed_inds"][0, 0] = self.num_embeddings
                ort_outs = ort_session.run(None, ort_inputs)

            try:            
                with self.assertRaises(Exception) as context4:
                    ort_inputs["embed_inds"][0, 0] = -1
                    ort_outs = ort_session.run(None, ort_inputs)
            except self.failureException as e:
                print("Warning: The onnxruntime does not raise exception when the input index is negative")
                warnings.warn("The onnxruntime does not raise exception when the input index is negative")
            


if __name__ == '__main__':
    unittest.main()
