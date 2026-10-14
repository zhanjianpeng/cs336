# scripts/generate.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import softmax


EOT_TOKEN = "<|endoftext|>"


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer LM."
    )

    # --------------------------------------------------------
    # Training artifacts
    # --------------------------------------------------------

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config.json saved by the training script.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
    )

    # --------------------------------------------------------
    # Tokenizer artifacts
    # --------------------------------------------------------

    parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
        help="Path to tokenizer vocabulary.",
    )

    parser.add_argument(
        "--merges",
        type=Path,
        required=True,
        help="Path to tokenizer BPE merges.",
    )

    # --------------------------------------------------------
    # Generation options
    # --------------------------------------------------------

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to continue.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of newly generated tokens.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Must be > 0.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold in (0, 1].",
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device, e.g. "cuda", "cpu", or "mps".',
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if args.max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be >= 0"
        )

    if args.temperature <= 0:
        raise ValueError(
            "temperature must be > 0"
        )

    if not (
        0.0
        < args.top_p
        <= 1.0
    ):
        raise ValueError(
            "top_p must satisfy 0 < top_p <= 1"
        )

    if not args.config.is_file():
        raise FileNotFoundError(
            f"Config file not found: {args.config}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {args.checkpoint}"
        )

    if not args.vocab.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found: {args.vocab}"
        )

    if not args.merges.is_file():
        raise FileNotFoundError(
            f"Merges file not found: {args.merges}"
        )

    return args


# ============================================================
# Load saved training config
# ============================================================

def load_training_config(
    config_path: Path,
) -> dict[str, Any]:

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        saved = json.load(
            f
        )

    # The training framework may store:
    #
    # {
    #     "config": {...},
    #     "num_parameters": ...
    # }
    #
    # Our previous train.py saved the TrainConfig
    # directly, so both formats are supported.

    if "config" in saved:
        return saved["config"]

    return saved


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer(
    vocab_path: Path,
    merges_path: Path,
) -> Tokenizer:

    tokenizer = Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=[
            EOT_TOKEN
        ],
    )

    return tokenizer


def get_eot_id(
    tokenizer: Tokenizer,
) -> int:
    """
    <|endoftext|> is a special token, so encoding it should
    produce exactly one token ID.
    """

    ids = tokenizer.encode(
        EOT_TOKEN
    )

    if len(ids) != 1:
        raise RuntimeError(
            f"{EOT_TOKEN!r} should encode to exactly one token, "
            f"but got {ids}"
        )

    return ids[0]


# ============================================================
# Model construction
# ============================================================

def build_model(
    config: dict[str, Any],
    device: str,
) -> TransformerLM:
    """
    Rebuild exactly the same architecture used during training.
    """

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],

        # Our previous TransformerLM constructor
        # uses the parameter name "theta".
        theta=config["rope_theta"],

        device=device,
    )

    return model


# ============================================================
# Checkpoint loading
# ============================================================

def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
) -> int | None:
    """
    Load model parameters for inference.

    Unlike training resume, generation does not need to
    restore AdamW state.

    Returns:
        checkpoint iteration if stored, otherwise None.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Earlier checkpoint format:
    #
    # checkpoint = {
    #     "model": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "iteration": iteration,
    # }
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model"]
    )

    return checkpoint.get(
        "iteration"
    )


# ============================================================
# Sampling
# ============================================================

def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> int:
    """
    Sample one token from next-token logits.

    Args:
        logits:
            Tensor of shape (vocab_size,).

        temperature:
            Positive sampling temperature.

        top_p:
            Nucleus-sampling threshold in (0, 1].

    Returns:
        Sampled token ID.
    """

    if logits.ndim != 1:
        raise ValueError(
            f"Expected 1-D logits, "
            f"got shape {tuple(logits.shape)}"
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0"
        )

    if not (
        0.0
        < top_p
        <= 1.0
    ):
        raise ValueError(
            "top_p must satisfy 0 < top_p <= 1"
        )


    # ========================================================
    # 1. Temperature scaling
    # ========================================================
    #
    # Assignment definition:
    #
    #     softmax(logits / temperature)
    #
    # Small temperature:
    #     sharper distribution
    #
    # Large temperature:
    #     flatter distribution

    scaled_logits = (
        logits
        / temperature
    )


    # ========================================================
    # 2. Convert to probabilities
    # ========================================================
    #
    # Output:
    #
    #     probs.shape == (V,)

    probs = softmax(
        scaled_logits,
        dim=-1,
    )


    # ========================================================
    # 3. Sort probabilities from largest to smallest
    # ========================================================
    #
    # sorted_probs:
    #
    #     probabilities sorted descending
    #
    # sorted_indices:
    #
    #     corresponding original vocabulary IDs

    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
    )


    # ========================================================
    # 4. Compute cumulative probabilities
    # ========================================================
    #
    # Example:
    #
    # sorted_probs:
    #
    #   [0.45, 0.30, 0.10, 0.05, ...]
    #
    # cumulative_probs:
    #
    #   [0.45, 0.75, 0.85, 0.90, ...]

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )


    # ========================================================
    # 5. Construct top-p prefix
    # ========================================================
    #
    # Keep the SMALLEST prefix whose cumulative probability
    # is >= top_p.
    #
    # IMPORTANT:
    #
    # The first token that causes cumulative probability
    # to cross top_p must still be INCLUDED.
    #
    # Read the previous cumulative value directly to avoid
    # cancellation at exact thresholds. top_p=1 keeps the
    # full distribution, including small positive probabilities.

    keep_mask = torch.ones_like(
        sorted_probs,
        dtype=torch.bool,
    )

    if top_p < 1.0:
        keep_mask[1:] = cumulative_probs[:-1] < top_p


    # ========================================================
    # 6. Remove tokens outside nucleus and renormalize
    # ========================================================

    filtered_probs = torch.where(
        keep_mask,
        sorted_probs,
        torch.zeros_like(
            sorted_probs
        ),
    )

    filtered_probs = (
        filtered_probs
        / torch.sum(
            filtered_probs
        )
    )


    # ========================================================
    # 7. Sample from the filtered distribution
    # ========================================================

    sampled_sorted_index = torch.multinomial(
        filtered_probs,
        num_samples=1,
    ).item()


    # ========================================================
    # 8. Map back to original vocabulary ID
    # ========================================================

    next_token_id = sorted_indices[
        sampled_sorted_index
    ].item()

    return int(
        next_token_id
    )


# ============================================================
# Autoregressive generation
# ============================================================

def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    context_length: int,
    device: str,
) -> list[int]:
    """
    Generate tokens autoregressively.

    Stops when:
        1. <|endoftext|> is sampled, or
        2. max_new_tokens tokens have been generated.

    The full history is retained for decoding, but only the
    most recent context_length tokens are passed to the model.
    """

    # ========================================================
    # 1. Encode prompt
    # ========================================================

    prompt_ids = tokenizer.encode(
        prompt
    )

    if len(prompt_ids) == 0:
        raise ValueError(
            "Prompt encoded to zero tokens. "
            "Please provide a non-empty prompt."
        )

    eot_id = get_eot_id(
        tokenizer
    )

    # Full token history.

    generated_ids = list(
        prompt_ids
    )


    # ========================================================
    # 2. Inference mode
    # ========================================================

    model.eval()

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            # =================================================
            # 3. Respect context-window limit
            # =================================================
            #
            # Suppose:
            #
            # context_length = 256
            # generated history = 400 tokens
            #
            # We keep all 400 tokens in generated_ids,
            # but the Transformer only receives:
            #
            #     generated_ids[-256:]
            #
            # This prevents exceeding the RoPE/model context
            # length while keeping complete output history.

            context_ids = generated_ids[
                -context_length:
            ]


            # =================================================
            # 4. Convert to batched tensor
            # =================================================
            #
            # context_ids:
            #     Python list length S
            #
            # model_input:
            #     shape (1, S)
            #
            # dtype:
            #     torch.long

            model_input = torch.tensor(
                context_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(
                0
            )


            # =================================================
            # 5. Forward
            # =================================================
            #
            # model_input:
            #     (1, S)
            #
            # logits:
            #     (1, S, V)

            logits = model(
                model_input
            )


            # =================================================
            # 6. Take only final-position logits
            # =================================================
            #
            # We need:
            #
            # P(next token | all current context)
            #
            # Shape:
            #
            #     (V,)

            next_token_logits = logits[
                0,
                -1,
                :,
            ]


            # =================================================
            # 7. Sample next token
            # =================================================

            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )


            # =================================================
            # 8. Stop at EOT
            # =================================================
            #
            # Here we do NOT append the EOT token to the final
            # output.

            if next_token_id == eot_id:
                break


            # =================================================
            # 9. Append and continue
            # =================================================

            generated_ids.append(
                next_token_id
            )


    return generated_ids


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()


    # ========================================================
    # Random seed
    # ========================================================

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )


    # ========================================================
    # 1. Load training configuration
    # ========================================================

    config = load_training_config(
        args.config
    )

    context_length = int(
        config["context_length"]
    )

    if context_length != 256:
        print(
            f"[warning] model context_length={context_length}; "
            "this script will use the context length saved in "
            "the training configuration."
        )


    # ========================================================
    # 2. Load tokenizer
    # ========================================================

    tokenizer = load_tokenizer(
        args.vocab,
        args.merges,
    )

    eot_id = get_eot_id(
        tokenizer
    )

    print(
        f"device: {args.device}"
    )

    print(
        f"context length: {context_length}"
    )

    print(
        f"EOT token id: {eot_id}"
    )


    # ========================================================
    # 3. Build model
    # ========================================================

    model = build_model(
        config,
        device=args.device,
    )


    # ========================================================
    # 4. Load checkpoint
    # ========================================================

    checkpoint_iteration = load_model_checkpoint(
        model,
        args.checkpoint,
        device=args.device,
    )

    if checkpoint_iteration is not None:

        print(
            f"loaded checkpoint from training step "
            f"{checkpoint_iteration}"
        )

    else:

        print(
            "loaded checkpoint"
        )


    # ========================================================
    # 5. Generate
    # ========================================================

    generated_ids = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=context_length,
        device=args.device,
    )


    # ========================================================
    # 6. Decode
    # ========================================================

    generated_text = tokenizer.decode(
        generated_ids
    )


    # ========================================================
    # 7. Output
    # ========================================================

    num_prompt_tokens = len(
        tokenizer.encode(
            args.prompt
        )
    )

    num_generated_tokens = (
        len(generated_ids)
        - num_prompt_tokens
    )

    print()

    print(
        "=" * 80
    )

    print(
        "GENERATION"
    )

    print(
        "=" * 80
    )

    print(
        generated_text
    )

    print(
        "=" * 80
    )

    print(
        f"prompt tokens: {num_prompt_tokens} | "
        f"generated tokens: {num_generated_tokens} | "
        f"temperature: {args.temperature} | "
        f"top_p: {args.top_p}"
    )


if __name__ == "__main__":
    main()
