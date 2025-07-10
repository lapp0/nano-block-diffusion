# Nano Block Diffusion

Lightweight codebase for training a [Block Diffusion](https://arxiv.org/abs/2503.09573) model from scratch in a few hours. Includes model definition, training loop, and inference script—designed as a foundation for further experimentation.

**Requirements:** CUDA 12.6, NVIDIA GPU with ≥24 GB VRAM - works with consumer hardware (e.g. RTX 3090) and server clusters (e.g. 8×H100).

## Block Diffusion
Executive summary of how Block Diffusion differs from a GPT-style transformer:
- **Block Diffusion:** Sample a noise level per `block_size` block, independently mask tokens at that probability, then reconstruct the entire block from its context in one shot rather than token-by-token.
- **Attention Rules:**
  - Within noisy block tokens attend to one another fully.
  - Noisy blocks attend to earlier, clean blocks.
  - Clean blocks follow standard causal masking.
- **Loss Function:** Apply cross-entropy only on masked tokens, weighting each block’s loss by the inverse of its noise level before averaging.

In commit [dcda272](https://github.com/lapp0/nano-block-diffusion/commit/dcda272db1606ac41623cce5f8dec7b1b8215f51), all core modifications required to transform nanoGPT into a Block Diffusion model are implemented.


## Training Instructions
**Clone & install dependencies**
```bash
git clone https://github.com/lapp0/nano-block-diffusion.git && cd nano-block-diffusion
pip install -r requirements.txt
pip install --pre torch==2.8.0.dev20250523+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
```

**Download training data**
```bash
python data/cached_finewebedu10B.py 10  # download input data
```

**Launch training on all GPUs**
```bash
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
```

## Training Records

| Implementation                                                   | 1xH100 Train Time | Train Tokens | Date       | Note                           |
|------------------------------------------------------------------|-------------------|--------------|------------|--------------------------------|
| Paper[1]                                                        |                   | 65,000M      | 05/17/2025 | Table 3, 30.60 PPL on **LM1B** |
| Nano Block Diffusion v0                                          |                   |              | 07/09/2025 | Original release               |
| [New Record](https://github.com/lapp0/nano-block-diffusion/pulls) |                   |              |            |                                |

PRs which improve the models training performance are encouraged. Block Diffusion models are new and underexplored.

**New Record Rules**
* **Parameter limit**: Use ≤ 162M parameters (including embeddings).
* **Target**: Achieve ≤ 3.44 cross-entropy loss on FineWebEdu validation set.
* **Data**: Must use FineWeb-Edu dataset. Sample order are fixed. Samples cannot be repeated. Sample size per batch may vary.
* **Objective:** Must retain the same objective function and retain 16 token denoising block.

## Notes

_Inspired by Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT).
Optimizer and architectural enhancements borrowed from [Muon](https://github.com/KellerJordan/Muon) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)._

### References

- [1]: [Arriola, M., Gokaslan, A., Chiu, J.T., Yang, Z., Qi, Z., Han, J., Sahoo, S.S., Kuleshov, V. (2025). Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models. arXiv preprint arXiv:2503.09573.](https://arxiv.org/abs/2503.09573)
- [2]: [Keller Jordan et al. *modded-nanogpt: Speedrunning the NanoGPT baseline*.](https://github.com/KellerJordan/modded-nanogpt/)
  - Note: many improvements in `model.py` are from modded-nanogpt, however, comments attributing the discovering author have been removed to keep the codebase clean. To view each improvements discovering author comment, see [modded-nanogpt's train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)



