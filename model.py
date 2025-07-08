from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask


@dataclass
class GPTConfig:
    vocab_size: int = 50_257
    bos_token_id: int = 50_256
    num_layers: int = 12
    num_heads: int = 6
    model_dim: int = 768
    max_seq_len: int = 131_072  # 2**17
    head_dim: int = 128
    intermediate_dim: int | None = None


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

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.c_q = CastedLinear(config.model_dim, config.model_dim)
        self.c_k = CastedLinear(config.model_dim, config.model_dim)
        self.c_v = CastedLinear(config.model_dim, config.model_dim)
        self.o_proj = CastedLinear(config.num_heads * config.head_dim, config.model_dim, zero_init=True)

        self.rotary = Rotary(config.head_dim, config.max_seq_len)
        self.lamb = nn.Parameter(torch.tensor(0.5))

        self.kernel_options = {
            "BLOCK_M": 64, "BLOCK_N": 64,
            "BLOCK_M1": 32, "BLOCK_N1": 64,
            "BLOCK_M2": 64, "BLOCK_N2": 32,
        }

    def forward(self, x: Tensor, v_residual: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        q = self.c_q(x).view(B, T, self.config.num_heads, -1)
        k = self.c_k(x).view(B, T, self.config.num_heads, -1)
        v = self.c_v(x).view(B, T, self.config.num_heads, -1)

        # norm and rotary
        q, k = self.rotary(norm(q)), self.rotary(norm(k))

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
    def __init__(self, config: GPTConfig):
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
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, v_residual: Tensor, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v_residual = self.attn(F.rms_norm(x, (x.size(-1),)), v_residual, block_mask)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v_residual


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        adj_vocab_size = ((config.vocab_size + 127) // 128) * 128  # out embed, round to nearest 128 for efficiency
        self.lm_head = CastedLinear(config.model_dim, adj_vocab_size, zero_init=True)

        assert len(self.blocks) % 2 == 0
        self.skip_w = nn.Parameter(torch.ones(len(self.blocks) // 2))

    def create_blockmask(self, input_seq: Tensor):
        docs = (input_seq == self.config.bos_token_id).cumsum(0)

        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask  # & window_mask

        S = len(input_seq)
        return create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

    def forward(self, input_seq: Tensor, target_seq: Tensor):
        assert input_seq.ndim == 1

        x = x0 = norm(self.embed(input_seq)[None])
        v_residual = None

        # U-net design
        block_mask = self.create_blockmask(input_seq)
        skip_conns, n = [], len(self.skip_w)
        for i, block in enumerate(self.blocks):
            if i >= n:
                x = x + self.skip_w[i - n] * skip_conns.pop()
            x, v_residual = block(x, v_residual, x0, block_mask)
            if i < n:
                skip_conns.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()

        # tanh softcapping
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')

        return loss
