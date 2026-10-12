import torch
from torch import nn

from cs336_basics.layers import RMSNorm, SwiGLU
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.layers import Embedding, Linear

#Transformer block: 1. RMSNorm -> 2. MultiHeadSelfAttention -> 3. RMSNorm -> 4. SwiGLU
class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.norm1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            use_rope=True,
        )

        self.norm2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        attention_update = self.attention(
            self.norm1(x),
            token_positions,
        )

        z = (
            x
            + attention_update
        )

        ffn_update = self.ffn(
            self.norm2(z)
        )

        output = (
            z
            + ffn_update
        )

        return output

#transformer language model: 1. Embedding -> 2. TransformerBlock -> 3. RMSNorm -> 4. Linear
class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )


    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        x = self.token_embedding(
            token_ids
        )

        seq_len = token_ids.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length {self.context_length}."
            )

        token_positions = torch.arange(
            seq_len,
            device=token_ids.device,
        )

        for layer in self.layers:

            x = layer(
                x,
                token_positions,
            )

        x = self.final_norm(
            x
        )

        logits = self.lm_head(
            x
        )

        return logits