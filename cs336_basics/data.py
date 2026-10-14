import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    num_tokens = len(dataset)

    num_valid_starts = (
        num_tokens
        - context_length
    )

    if num_valid_starts <= 0:
        raise ValueError(
            "dataset must contain more tokens "
            "than context_length."
        )

    starts = np.random.randint(
        low=0,
        high=num_valid_starts,
        size=batch_size,
    )

    inputs = np.stack(
        [
            dataset[
                s : s + context_length
            ]
            for s in starts
        ]
    )

    targets = np.stack(
        [
            dataset[
                s + 1 : s + context_length + 1
            ]
            for s in starts
        ]
    )

    inputs = torch.tensor(
        inputs,
        dtype=torch.long,
    )

    targets = torch.tensor(
        targets,
        dtype=torch.long,
    )

    inputs = inputs.to(
        device
    )

    targets = targets.to(
        device
    )


    return inputs, targets