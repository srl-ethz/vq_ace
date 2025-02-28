# VQ-ACE: Efficient Policy Search for Dexterous Robotic Manipulation via Action Chunking Embedding

The code framework is adapted from [srl_il](https://github.com/srl-ethz/srl_il)

## Get started

The following is tested on Ubuntu22.04

```bash
conda create -n vq_ace python=3.11 # or other versions of python
conda activate vq_ace

git clone git@github.com:srl-ethz/vq_ace.git
cd vq_ace
pip install -e .
```

## Training


Train the action chunk embeddings without vector quantization.

```bash
python3 scripts/run_pipeline.py --config-name=train_embed_act
```

Train the action chunk embeddings with vector quantization.

```bash
python3 scripts/run_pipeline.py --config-name=train_embed_vq_act
```

Train the action chunk embeddings with vector quantization, but without conditions

```bash
python3 scripts/run_pipeline.py --config-name=train_embed_vq_act
```


## 🙏Acknowledgement
- The robomimic tasks and observation encoders are adapted from [Robomimic](https://github.com/ARISE-Initiative/robomimic)
- The linear normalizer implementation is adapted from [diffusion policy](https://github.com/real-stanford/diffusion_policy)
- The vector quantize implementation is adapted from [vq_bet_officia](https://github.com/jayLEE0301/vq_bet_official)


