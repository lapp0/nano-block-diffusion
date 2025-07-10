from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask


create_block_mask = torch.compile(create_block_mask, dynamic=False)
flex_attention = torch.compile(flex_attention, dynamic=False)


@dataclass
class BlockGPTConfig:
    vocab_size: int = 50_258
    bos_id: int = 50_256
    mask_id: int = 50_257
    num_layers: int = 12
    num_heads: int = 6
    model_dim: int = 768
    max_seq_len: int = int(2**16)
    head_dim: int = 128
    intermediate_dim: int | None = None
    diffusion_block_size: int = 16
    t_lower: float = 0.3
    t_upper: float = 0.8


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, zero_init=False):
        self._zero_init = zero_init
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        if self._zero_init:
            self.weight.detach().zero_()
        else:
            with torch.no_grad():
                bound = (3 ** 0.5) * 0.5 * (self.in_features ** -0.5)
                self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor, pos_id: Tensor):
        assert self.cos.size(0) > pos_id.max()
        cos, sin = self.cos[pos_id, None, :], self.sin[pos_id, None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BlockGPTConfig):
        super().__init__()
        self.config = config

        self.c_q = CastedLinear(config.model_dim, config.model_dim)
        self.c_k = CastedLinear(config.model_dim, config.model_dim)
        self.c_v = CastedLinear(config.model_dim, config.model_dim)
        self.o_proj = CastedLinear(config.num_heads * config.head_dim, config.model_dim)

        self.rotary = Rotary(config.head_dim, config.max_seq_len)
        self.lamb = nn.Parameter(torch.tensor(0.5))

        self.kernel_options = {
            "BLOCK_M": 32, "BLOCK_N": 32,
            "BLOCK_M1": 16, "BLOCK_N1": 32,
            "BLOCK_M2": 32, "BLOCK_N2": 16,
        }

    def forward(self, x: Tensor, v_residual: Tensor | None, pos_id: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        q = self.c_q(x).view(B, T, self.config.num_heads, -1)
        k = self.c_k(x).view(B, T, self.config.num_heads, -1)
        v = self.c_v(x).view(B, T, self.config.num_heads, -1)

        # norm and rotary
        q, k = self.rotary(norm(q), pos_id), self.rotary(norm(k), pos_id)

        # initialize residual on first pass
        v_residual = v_residual if v_residual is not None else v
        v = (1 - self.lamb) * v + self.lamb * v_residual.view_as(v)

        # flex‐attention & re-project
        y = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            block_mask=block_mask,
            kernel_options=self.kernel_options
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, -1)
        return self.o_proj(y), v_residual


class MLP(nn.Module):
    def __init__(self, config: BlockGPTConfig):
        super().__init__()
        intermediate_dim = config.intermediate_dim or 4 * config.model_dim
        self.in_proj = CastedLinear(config.model_dim, intermediate_dim)
        self.out_proj = CastedLinear(intermediate_dim, config.model_dim, zero_init=True)

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = F.relu(x).square()
        x = self.out_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: BlockGPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, v_residual: Tensor, x0: Tensor, pos_id: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v_residual = self.attn(norm(x), v_residual, pos_id, block_mask)
        x = x + x1
        x = x + self.mlp(norm(x))
        return x, v_residual


class BlockGPT(nn.Module):
    def __init__(self, config: BlockGPTConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        adj_vocab_size = ((config.vocab_size + 127) // 128) * 128  # out embed, round to nearest 128 for efficiency
        self.lm_head = CastedLinear(config.model_dim, adj_vocab_size, zero_init=True)

        assert len(self.blocks) % 2 == 0
        self.skip_w = nn.Parameter(torch.ones(len(self.blocks) // 2))

    def create_blockmask(self, doc_id: Tensor, pos_id: Tensor):
        """BlockMask for attn rules from https://arxiv.org/pdf/2503.09573 section 3.1"""
        L = len(doc_id)

        block_id = pos_id // self.config.diffusion_block_size + doc_id * L
        block_id = torch.cumsum(block_id != block_id.roll(1, 0), 0) - 1

        block_id, doc_id = block_id.repeat(2), doc_id.repeat(2)
        noisy = torch.arange(2 * L, device=doc_id.device) < L

        def block_diffusion_mask(b, h, q, kv):
            # mask from section 3.1 of https://arxiv.org/pdf/2503.09573
            blk_q, blk_kv = block_id[q], block_id[kv]

            bd = (blk_q == blk_kv) & (noisy[q] == noisy[kv])  # Block Diagonal
            obc = (blk_q > blk_kv) & noisy[q] & (~noisy[kv])  # Offset Block Causal
            bc = (blk_q >= blk_kv) & (~noisy[q]) & (~noisy[kv])  # Block Causal

            same_doc = doc_id[q] == doc_id[kv]
            return same_doc & (bd | obc | bc)

        S = 2 * L
        return create_block_mask(block_diffusion_mask, None, None, S, S)

    def forward(self, input_seq: Tensor):
        assert input_seq.ndim == 1

        # construct attention rules & block mask
        doc_id = (input_seq == self.config.bos_id).cumsum(0)
        p = torch.arange(input_seq.size(0), device=input_seq.device)
        pos_id = p - torch.where(input_seq == self.config.bos_id, p, -1).cummax(0).values
        block_mask = self.create_blockmask(doc_id, pos_id)

        # Apply noise to sequence
        noise_range = (self.config.t_lower, self.config.t_upper) if self.training else (0.0, 1.0)
        rand = torch.rand_like(input_seq, dtype=torch.float32)
        t = torch.empty_like(rand).uniform_(*noise_range)[doc_id]
        noisy_seq = input_seq.masked_fill(rand >= (1 - t), self.config.mask_id)

        # Concat noisy + clean into seq and repeat pos_ids
        seq = torch.cat([noisy_seq, input_seq], dim=0)
        pos_id = pos_id.repeat(2)

        # Embedding & U-net backbone forward
        x = x0 = norm(self.embed(seq)[None])
        v_residual = None

        skip_conns, n = [], len(self.skip_w)
        for i, block in enumerate(self.blocks):
            if i >= n:
                x = x + self.skip_w[i - n] * skip_conns.pop()
            x, v_residual = block(x, v_residual, x0, pos_id, block_mask)
            if i < n:
                skip_conns.append(x)

        x = x[:, :input_seq.size(0)]  # Get logits for noisy tokens only
        x = norm(x)
        logits = self.lm_head(x).float()

        # Get loss for masked tokens
        mask = (noisy_seq == self.config.mask_id)
        targets = torch.where(mask, input_seq, torch.full_like(input_seq, -100))
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        if self.training:
            weights = (1.0 / (t + 1e-4)).type_as(logits)
            return (losses * weights * mask).sum() / mask.sum()
        else:
            return (losses * mask).sum() / mask.sum()
