import torch

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:

    max_value = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    shifted_x = (
        x - max_value
    )

    exp_x = torch.exp(
        shifted_x
    )

    sum_exp = torch.sum(
        exp_x,
        dim=dim,
        keepdim=True,
    )

    output = (
        exp_x / sum_exp
    )

    return output