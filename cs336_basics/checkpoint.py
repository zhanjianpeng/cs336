import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:

    model_state = (
        model.state_dict()
    )

    optimizer_state = (
        optimizer.state_dict()
    )

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer_state,
        "iteration": iteration,
    }

    torch.save(
        checkpoint,
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    checkpoint = torch.load(
        src
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    iteration = checkpoint[
        "iteration"
    ]

    return iteration