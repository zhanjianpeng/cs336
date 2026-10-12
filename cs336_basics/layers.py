import math

import torch
from torch import nn

#Linear layer
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # 在 [-3 * std, 3 * std] 范围初始化 weight。
        std = math.sqrt(
            2 / (in_features + out_features)
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output = x @ self.weight.T

        return output

#Embedding layer
class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )


    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        output = self.weight[token_ids]

        return output

#RMSNorm layer
class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        in_dtype = x.dtype

        x = x.to(torch.float32)

        mean_square = torch.mean(
            x ** 2,
            dim=-1,
            keepdim=True,
        )

        rms = torch.sqrt(
            mean_square + self.eps
        )

        normalized = x / rms

        result = normalized * self.gain

        return result.to(in_dtype)

#silu
def silu(
    x: torch.Tensor,
) -> torch.Tensor:

    return x * torch.sigmoid(x)

# SwiGLU layer
class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model

        if d_ff is None:

            approximate_d_ff = (
                (8 / 3) * d_model
            )

            d_ff = 64 * round(
                approximate_d_ff / 64
            )

            d_ff = max(
                64,
                d_ff,
            )

        self.d_ff = d_ff

        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate_input = self.w1(x)

        activated = silu(
            gate_input
        )

        value = self.w3(x)

        gated = (
            activated
            * value
        )

        output = self.w2(
            gated
        )

        return output

#RoPE layer
class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        if d_k % 2 != 0:
            raise ValueError(
                "d_k must be even for RoPE."
            )

        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        frequencies = (
            theta
            ** (
                -dimension_indices / d_k
            )
        )

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles = (
            positions[:, None]
            * frequencies[None, :]
        )

        cos = torch.cos(
            angles
        )

        sin = torch.sin(
            angles
        )

        self.register_buffer(
            "cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin",
            sin,
            persistent=False,
        )


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        cos = self.cos[
            token_positions
        ]
        sin = self.sin[
            token_positions
        ]

        cos = cos.to(
            dtype=x.dtype
        )
        sin = sin.to(
            dtype=x.dtype
        )

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )
        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.stack(
            (
                rotated_even,
                rotated_odd,
            ),
            dim=-1,
        )

        output = output.reshape(
            *x.shape[:-1],
            self.d_k,
        )

        return output