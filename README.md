# Nano Block Diffusion

Simple codebase for training a [Block Diffusion](https://arxiv.org/abs/2503.09573) model from scratch in a few hours.

We provide a trainer which runs on consumer hardware (e.g. 1x3090) and server clusters (e.g. 8xH100).


## Overview
This repo implements a simple Block Diffusion model, trainer, and inference script. This serves as a starting point for researching improvements integration into existing models.

Executive summary of the core difference between a standard autoregressive "GPT-style" transformer and a Block Diffusion Transformer:
- **Block Diffusion:** Rather than predicting just the next token, the model masks out the next `B` tokens in one go and learns to reconstruct entire blocks from their surrounding context.
- **Attention Rules:** Full attention within each block; noisy blocks may causally attend to preceding clean blocks; clean blocks obey standard causal masking.
- **Loss Function:** Cross-entropy on masked tokens. Each loss term is weighted by the inverse of its block’s noise level and averaged.

In this simple [commit](https://github.com/lapp0/nano-block-diffusion/commit/dcda272db1606ac41623cce5f8dec7b1b8215f51) we apply all necessary changes to convert a GPT model into a Block Diffusion model.

_Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and incorporates improved [optimizers](https://github.com/KellerJordan/Muon) model architecture and from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)._


## Training Instructions
**Setup** code, packages, and data.
```bash
git clone https://github.com/lapp0/nano-block-diffusion.git && cd nano-block-diffusion
pip install -r requirements.txt
pip install --pre torch==2.8.0.dev20250523+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_finewebedu10B.py 10  # download input data
```

**Run** using all available GPUs
```bash
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
```

## Training Records

| Implementation                                                   | 1xH100 Train Time | Train Tokens | Date       | Note                           |
|------------------------------------------------------------------|-------------------|--------------|------------|--------------------------------|
| Paper[^1]                                                        |                   | 65,000M      | 05/17/2025 | Table 3, 30.60 PPL on **LM1B** |
| Nano Block Diffusion v0                                          |                   |              | 07/09/2025 | Original release               |
| [New Record](https://github.com/lapp0/nano-block-diffusion/pulls |                   |              |            |                                |

PRs which improve the models training performance are encouraged. Block Diffusion models are new and underexplored.

**New Record Rules**
* **Parameter limit**: Use ≤ 162M parameters (including embeddings).
* **Target**: Achieve ≤ 3.44 cross-entropy loss on FineWebEdu validation set.
* **Data**: Must use FineWeb dataset. Sample order are fixed. Samples cannot be repeated. Sample size per batch may vary.
* **Objective:** Must retain the same objective function and retain 16 token denoising block.

## Citations

[^1]: [Arriola, M., Gokaslan, A., Chiu, J.T., Yang, Z., Qi, Z., Han, J., Sahoo, S.S., Kuleshov, V. (2025). Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models. arXiv preprint arXiv:2503.09573.](https://arxiv.org/abs/2503.09573)
[^2]: [Keller Jordan et al. *modded-nanogpt: Speedrunning the NanoGPT baseline*.](https://github.com/KellerJordan/modded-nanogpt/)
- Note: many improvements in `model.py` are from modded-nanogpt, however, comments attributing the discovering author have been removed to keep the codebase clean. To view each improvements discovering author comment, see [modded-nanogpt's train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)
