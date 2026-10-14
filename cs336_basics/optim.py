import math

import torch
from collections.abc import Callable, Iterable
from typing import Optional


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):

        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError(
                f"Invalid beta1: {beta1}"
            )

        if not 0 <= beta2 < 1:
            raise ValueError(
                f"Invalid beta2: {beta2}"
            )

        if eps < 0:
            raise ValueError(
                f"Invalid epsilon: {eps}"
            )

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight decay: {weight_decay}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(
            params,
            defaults,
        )


    def step(
        self,
        closure: Optional[Callable] = None,
    ):
        loss = (
            None
            if closure is None
            else closure()
        )

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:

                    state["t"] = 0

                    state["m"] = torch.zeros_like(
                        p
                    )

                    state["v"] = torch.zeros_like(
                        p
                    )

                m = state["m"]
                v = state["v"]

                state["t"] += 1
                t = state["t"]

                p.data -= (
                    lr
                    * weight_decay
                    * p.data
                )

                m.mul_(
                    beta1
                )

                m.add_(
                    grad,
                    alpha=1 - beta1,
                )

                v.mul_(
                    beta2
                )

                v.add_(
                    grad * grad,
                    alpha=1 - beta2,
                )

                adjusted_lr = (
                    lr
                    * math.sqrt(
                        1 - beta2 ** t
                    )
                    / (
                        1 - beta1 ** t
                    )
                )

                p.data -= (
                    adjusted_lr
                    * m
                    / (
                        torch.sqrt(v)
                        + eps
                    )
                )

        return loss


def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
   
    if t < warmup_iters:

        warmup_fraction = (
            t / warmup_iters
        )

        lr = (
            warmup_fraction
            * max_learning_rate
        )

        return lr

    if t <= cosine_cycle_iters:

        progress = (
            (t - warmup_iters)
            / (
                cosine_cycle_iters
                - warmup_iters
            )
        )

        cosine_coefficient = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        lr = (
            min_learning_rate
            + cosine_coefficient
            * (
                max_learning_rate
                - min_learning_rate
            )
        )

        return lr

    return min_learning_rate