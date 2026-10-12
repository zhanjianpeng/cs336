import math

import torch

from torch import nn

from cs336_basics.nn_utils import softmax
from cs336_basics.layers import Linear, RotaryPositionalEmbedding

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    scores = (
        q
        @ k.transpose(-2, -1)
    )

    d_k = q.shape[-1]

    scores = (
        scores
        / math.sqrt(d_k)
    )

    if mask is not None:

        mask = mask.to(
            device=scores.device,
            dtype=torch.bool,
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    attention_weights = softmax(
        scores,
        dim=-1,
    )

    output = (
        attention_weights
        @ v
    )

    return output

#多头因果注意力
class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
        use_rope: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_head = (
            d_model // num_heads
        )

        if use_rope and self.d_head % 2 != 0:
            raise ValueError(
                "d_head must be even for RoPE."
            )

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.rope = None

        if use_rope:
            if theta is None or max_seq_len is None:
                raise ValueError(
                    "theta and max_seq_len are required when use_rope=True."
                )

            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )

        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        head_shape = (
            *batch_shape,
            seq_len,
            self.num_heads,
            self.d_head,
        )

        q = self.q_proj(x).reshape(head_shape).transpose(-3, -2)
        k = self.k_proj(x).reshape(head_shape).transpose(-3, -2)
        v = self.v_proj(x).reshape(head_shape).transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                )

            rope_positions = token_positions.to(
                device=x.device,
            ).unsqueeze(-2)

            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        attention_output = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=causal_mask,
        )

        concatenated = attention_output.transpose(
            -3, -2,
        ).reshape(
            *batch_shape,
            seq_len,
            self.d_model,
        )

        return self.output_proj(concatenated)