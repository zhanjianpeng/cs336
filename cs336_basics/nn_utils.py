from collections.abc import Iterable

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

#交叉熵
def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    max_logits = torch.max(
        logits,
        dim=-1,
        keepdim=True,
    ).values

    shifted_logits = (
        logits - max_logits
    )

    log_sum_exp = torch.log(
        torch.sum(
            torch.exp(
                shifted_logits
            ),
            dim=-1,
        )
    )

    correct_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    per_token_loss = (
        log_sum_exp
        - correct_logits
    )

    loss = torch.mean(
        per_token_loss
    )

    return loss


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> None:

    eps = 1e-6

    grads = []

    for p in parameters:
        if p.grad is None:
            continue

        grads.append(p.grad)


    if len(grads) == 0:
        return

    total_squared_norm = torch.zeros(
        (),
        device=grads[0].device,
        dtype=grads[0].dtype,
    )

    for grad in grads:

        total_squared_norm += torch.sum(
            grad ** 2
        )

    global_norm = torch.sqrt(
        total_squared_norm
    )

    if global_norm <= max_norm:
        return

    scale = (
        max_norm
        / (
            global_norm
            + eps
        )
    )

    for grad in grads:

        grad.mul_(
            scale
        )
