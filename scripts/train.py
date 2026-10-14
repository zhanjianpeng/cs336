from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


from cs336_basics.transformer import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optim import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # -------------------------
    # Data
    # -------------------------
    train_data_path: str
    valid_data_path: str
    output_dir: str

    # -------------------------
    # Model
    # -------------------------
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    # -------------------------
    # Optimization
    # -------------------------
    batch_size: int
    max_steps: int

    max_lr: float
    min_lr: float
    warmup_steps: int
    cosine_cycle_steps: int

    beta1: float
    beta2: float
    eps: float
    weight_decay: float

    max_grad_norm: float

    # -------------------------
    # Evaluation / logging
    # -------------------------
    log_interval: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int

    # -------------------------
    # Runtime
    # -------------------------
    device: str
    seed: int

    # None means start fresh.
    resume_from: str | None = None


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model."
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--valid-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--d-ff",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--rope-theta",
        type=float,
        required=True,
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-lr",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--cosine-cycle-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--eps",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        required=True,
    )

    # --------------------------------------------------------
    # Evaluation / logging
    # --------------------------------------------------------

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    config = TrainConfig(
        train_data_path=args.train_data,
        valid_data_path=args.valid_data,
        output_dir=args.output_dir,

        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,

        batch_size=args.batch_size,
        max_steps=args.max_steps,

        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        cosine_cycle_steps=args.cosine_cycle_steps,

        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,

        max_grad_norm=args.max_grad_norm,

        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,

        device=args.device,
        seed=args.seed,

        resume_from=args.resume_from,
    )

    validate_config(
        config
    )

    return config


# ============================================================
# Configuration validation
# ============================================================

def validate_config(
    config: TrainConfig,
) -> None:

    if config.vocab_size <= 0:
        raise ValueError(
            "vocab_size must be positive."
        )

    if config.context_length <= 0:
        raise ValueError(
            "context_length must be positive."
        )

    if config.d_model <= 0:
        raise ValueError(
            "d_model must be positive."
        )

    if config.num_layers <= 0:
        raise ValueError(
            "num_layers must be positive."
        )

    if config.num_heads <= 0:
        raise ValueError(
            "num_heads must be positive."
        )

    if (
        config.d_model
        % config.num_heads
        != 0
    ):
        raise ValueError(
            "d_model must be divisible "
            "by num_heads."
        )

    d_head = (
        config.d_model
        // config.num_heads
    )

    if d_head % 2 != 0:
        raise ValueError(
            "d_head must be even for RoPE."
        )

    if config.d_ff <= 0:
        raise ValueError(
            "d_ff must be positive."
        )

    if config.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if config.max_steps <= 0:
        raise ValueError(
            "max_steps must be positive."
        )

    if config.max_lr < 0:
        raise ValueError(
            "max_lr must be non-negative."
        )

    if config.min_lr < 0:
        raise ValueError(
            "min_lr must be non-negative."
        )

    if config.min_lr > config.max_lr:
        raise ValueError(
            "min_lr must not exceed max_lr."
        )

    if config.warmup_steps < 0:
        raise ValueError(
            "warmup_steps must be non-negative."
        )

    if (
        config.cosine_cycle_steps
        <= config.warmup_steps
    ):
        raise ValueError(
            "cosine_cycle_steps must be greater "
            "than warmup_steps."
        )

    if not 0 <= config.beta1 < 1:
        raise ValueError(
            "beta1 must satisfy 0 <= beta1 < 1."
        )

    if not 0 <= config.beta2 < 1:
        raise ValueError(
            "beta2 must satisfy 0 <= beta2 < 1."
        )

    if config.eps < 0:
        raise ValueError(
            "eps must be non-negative."
        )

    if config.weight_decay < 0:
        raise ValueError(
            "weight_decay must be non-negative."
        )

    if config.max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be positive."
        )

    if config.log_interval <= 0:
        raise ValueError(
            "log_interval must be positive."
        )

    if config.eval_interval <= 0:
        raise ValueError(
            "eval_interval must be positive."
        )

    if config.eval_batches <= 0:
        raise ValueError(
            "eval_batches must be positive."
        )

    if config.checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be positive."
        )


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(
    path: str,
    context_length: int,
) -> np.ndarray:

    dataset_path = Path(
        path
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # Memory-mapped loading:
    #
    # the entire dataset does NOT need to be
    # copied into RAM.

    dataset = np.load(
        dataset_path,
        mmap_mode="r",
    )

    if dataset.ndim != 1:
        raise ValueError(
            f"Expected a 1D token array, "
            f"got shape {dataset.shape} "
            f"for {dataset_path}."
        )

    if not np.issubdtype(
        dataset.dtype,
        np.integer,
    ):
        raise TypeError(
            f"Expected integer token IDs, "
            f"got dtype {dataset.dtype}."
        )

    if len(dataset) <= context_length:
        raise ValueError(
            f"Dataset {dataset_path} is too short "
            f"for context_length={context_length}."
        )

    return dataset


# ============================================================
# Model / optimizer construction
# ============================================================

def build_model(
    config: TrainConfig,
) -> TransformerLM:

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        theta=config.rope_theta,
        device=config.device,
    )

    return model


def build_optimizer(
    model: TransformerLM,
    config: TrainConfig,
) -> AdamW:

    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(
            config.beta1,
            config.beta2,
        ),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    return optimizer


# ============================================================
# Learning-rate update
# ============================================================

def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:

    for group in optimizer.param_groups:
        group["lr"] = lr


# ============================================================
# Validation
# ============================================================

def evaluate(
    model: TransformerLM,
    valid_data: np.ndarray,
    config: TrainConfig,
) -> float:
    """
    Estimate validation loss over eval_batches
    independently sampled batches.
    """

    was_training = model.training

    model.eval()

    total_loss = torch.zeros(
        (),
        device=config.device,
        dtype=torch.float32,
    )

    with torch.no_grad():

        for _ in range(
            config.eval_batches
        ):

            inputs, targets = get_batch(
                dataset=valid_data,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=config.device,
            )

            logits = model(
                inputs
            )

            loss = cross_entropy(
                logits,
                targets,
            )

            total_loss += loss.detach()

    average_loss = (
        total_loss
        / config.eval_batches
    )

    if was_training:
        model.train()

    return average_loss.item()


# ============================================================
# Save training configuration
# ============================================================

def save_config(
    config: TrainConfig,
    output_dir: Path,
) -> None:

    config_path = (
        output_dir
        / "train_config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            asdict(config),
            f,
            indent=2,
        )


# ============================================================
# Training
# ============================================================

def train(
    config: TrainConfig,
) -> None:

    # --------------------------------------------------------
    # 1. Runtime setup
    # --------------------------------------------------------

    set_seed(
        config.seed
    )

    output_dir = Path(
        config.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config(
        config,
        output_dir,
    )

    print(
        f"Device: {config.device}"
    )


    # --------------------------------------------------------
    # 2. Load datasets using mmap
    # --------------------------------------------------------

    train_data = load_dataset(
        config.train_data_path,
        config.context_length,
    )

    valid_data = load_dataset(
        config.valid_data_path,
        config.context_length,
    )

    print(
        f"Train tokens: {len(train_data):,}"
    )

    print(
        f"Valid tokens: {len(valid_data):,}"
    )


    # --------------------------------------------------------
    # 3. Build model
    # --------------------------------------------------------

    model = build_model(
        config
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )


    # --------------------------------------------------------
    # 4. Build optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model,
        config,
    )


    # --------------------------------------------------------
    # 5. Optionally resume from checkpoint
    # --------------------------------------------------------
    #
    # Convention used here:
    #
    # iteration means:
    #
    #   number of optimization steps already completed
    #
    # Therefore if checkpoint contains iteration=1000,
    # the next training step is step 1000.
    # --------------------------------------------------------

    start_step = 0

    if config.resume_from is not None:

        resume_path = Path(
            config.resume_from
        )

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{resume_path}"
            )

        start_step = load_checkpoint(
            src=resume_path,
            model=model,
            optimizer=optimizer,
        )

        if start_step > config.max_steps:
            raise ValueError(
                f"Checkpoint has {start_step} completed steps, "
                f"but max_steps is {config.max_steps}. "
                "max_steps is the total training step limit; "
                "increase it to resume training."
            )

        print(
            f"Resumed from checkpoint: "
            f"{resume_path}"
        )

        print(
            f"Starting from step: "
            f"{start_step}"
        )


    # --------------------------------------------------------
    # 6. Training loop
    # --------------------------------------------------------

    model.train()

    running_loss = torch.zeros(
        (),
        device=config.device,
        dtype=torch.float32,
    )

    running_steps = 0

    last_log_step = (
        start_step
    )

    last_log_time = (
        time.perf_counter()
    )

    training_start_time = (
        last_log_time
    )


    for step in range(
        start_step,
        config.max_steps,
    ):

        # ----------------------------------------------------
        # A. Learning-rate schedule
        # ----------------------------------------------------

        lr = get_lr_cosine_schedule(
            t=step,
            max_learning_rate=config.max_lr,
            min_learning_rate=config.min_lr,
            warmup_iters=config.warmup_steps,
            cosine_cycle_iters=config.cosine_cycle_steps,
        )

        set_learning_rate(
            optimizer,
            lr,
        )


        # ----------------------------------------------------
        # B. Sample training batch
        # ----------------------------------------------------

        inputs, targets = get_batch(
            dataset=train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )


        # ----------------------------------------------------
        # C. Clear gradients from previous iteration
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # D. Forward pass
        # ----------------------------------------------------

        logits = model(
            inputs
        )


        # ----------------------------------------------------
        # E. Cross-entropy loss
        # ----------------------------------------------------

        loss = cross_entropy(
            logits,
            targets,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss encountered "
                f"at step {step}: {loss.item()}"
            )


        # ----------------------------------------------------
        # F. Backward pass
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # G. Gradient clipping
        #
        # Must happen AFTER backward
        # and BEFORE optimizer.step().
        # ----------------------------------------------------

        gradient_clipping(
            model.parameters(),
            max_norm=config.max_grad_norm,
        )


        # ----------------------------------------------------
        # H. AdamW update
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # step is zero-indexed.
        #
        # completed_step means the number of updates
        # that have actually been completed.
        # ----------------------------------------------------

        completed_step = (
            step + 1
        )

        running_loss += (
            loss.detach()
        )

        running_steps += 1


        # ----------------------------------------------------
        # I. Training logging
        # ----------------------------------------------------

        if (
            completed_step
            % config.log_interval
            == 0
        ):

            average_train_loss = (
                running_loss
                / running_steps
            ).item()

            now = time.perf_counter()

            elapsed = (
                now
                - last_log_time
            )

            steps_since_log = (
                completed_step
                - last_log_step
            )

            tokens_since_log = (
                steps_since_log
                * config.batch_size
                * config.context_length
            )

            tokens_per_second = (
                tokens_since_log
                / elapsed
            )

            print(
                json.dumps(
                    {
                        "step": completed_step,
                        "train_loss": average_train_loss,
                        "lr": lr,
                        "tokens_per_second": tokens_per_second,
                    }
                )
            )

            running_loss.zero_()

            running_steps = 0

            last_log_step = (
                completed_step
            )

            last_log_time = (
                now
            )


        # ----------------------------------------------------
        # J. Validation
        # ----------------------------------------------------

        if (
            completed_step
            % config.eval_interval
            == 0
        ):

            validation_loss = evaluate(
                model,
                valid_data,
                config,
            )

            print(
                json.dumps(
                    {
                        "step": completed_step,
                        "valid_loss": validation_loss,
                    }
                )
            )


        # ----------------------------------------------------
        # K. Periodic checkpoint
        # ----------------------------------------------------

        if (
            completed_step
            % config.checkpoint_interval
            == 0
        ):

            checkpoint_path = (
                output_dir
                / (
                    f"checkpoint_"
                    f"{completed_step:07d}.pt"
                )
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_step,
                out=checkpoint_path,
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_path}"
            )


    # --------------------------------------------------------
    # 7. Final validation
    # --------------------------------------------------------

    final_validation_loss = evaluate(
        model,
        valid_data,
        config,
    )

    print(
        json.dumps(
            {
                "step": config.max_steps,
                "final_valid_loss": (
                    final_validation_loss
                ),
            }
        )
    )


    # --------------------------------------------------------
    # 8. Final checkpoint
    # --------------------------------------------------------

    final_checkpoint_path = (
        output_dir
        / "checkpoint_final.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=config.max_steps,
        out=final_checkpoint_path,
    )


    # --------------------------------------------------------
    # 9. Final statistics
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - training_start_time
    )

    total_completed_steps = (
        config.max_steps
        - start_step
    )

    total_tokens = (
        total_completed_steps
        * config.batch_size
        * config.context_length
    )

    print(
        f"Training complete."
    )

    print(
        f"Final checkpoint: "
        f"{final_checkpoint_path}"
    )

    print(
        f"Training time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Tokens processed this run: "
        f"{total_tokens:,}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    config = parse_args()

    train(
        config
    )


if __name__ == "__main__":
    main()
