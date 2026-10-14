from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from cs336_basics.tokenizer import Tokenizer


EOT_TOKEN = "<|endoftext|>"


@dataclass
class Config:
    input_path: Path
    vocab_path: Path
    merges_path: Path
    output_dir: Path

    valid_ratio: float = 0.1
    seed: int = 42


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Preprocess TinyStories into train.npy / valid.npy"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--merges",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()


    if not 0 < args.valid_ratio < 1:
        parser.error(
            "--valid-ratio must satisfy 0 < valid_ratio < 1"
        )

    if not args.input.is_file():
        parser.error(
            f"Input file does not exist: {args.input}"
        )

    if not args.vocab.is_file():
        parser.error(
            f"Vocabulary file does not exist: {args.vocab}"
        )

    if not args.merges.is_file():
        parser.error(
            f"Merges file does not exist: {args.merges}"
        )

    return Config(
        input_path=args.input,
        vocab_path=args.vocab,
        merges_path=args.merges,
        output_dir=args.output_dir,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )


def load_tokenizer(
    config: Config,
) -> Tokenizer:

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(config.vocab_path),
        merges_filepath=str(config.merges_path),
        special_tokens=[EOT_TOKEN],
    )

    return tokenizer


def get_vocab_size(
    tokenizer: Tokenizer,
) -> int:

    vocab_size = len(
        tokenizer.vocab
    )

    return vocab_size


def load_stories(
    input_path: Path,
) -> list[str]:


    text = input_path.read_text(
        encoding="utf-8"
    )


    stories = text.split(
        EOT_TOKEN
    )


    while (
        len(stories) > 0
        and stories[-1].strip() == ""
    ):
        stories.pop()

    if len(stories) == 0:
        raise ValueError(
            "No stories were found in the input corpus."
        )

    return stories


def split_stories(
    stories: Sequence[str],
    valid_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:


    num_stories = len(stories)

    if num_stories < 2:
        raise ValueError(
            "At least two stories are required "
            "for a train/validation split."
        )

    rng = np.random.default_rng(
        seed
    )


    indices = rng.permutation(
        num_stories
    )


    num_valid = round(
        num_stories
        * valid_ratio
    )

    num_valid = max(
        1,
        num_valid,
    )

    num_valid = min(
        num_stories - 1,
        num_valid,
    )


    valid_indices = indices[
        :num_valid
    ]

    train_indices = indices[
        num_valid:
    ]


    train_stories = [
        stories[i]
        for i in train_indices
    ]

    valid_stories = [
        stories[i]
        for i in valid_indices
    ]

    return train_stories, valid_stories


def choose_token_dtype(
    vocab_size: int,
) -> np.dtype:


    if vocab_size <= 0:
        raise ValueError(
            "vocab_size must be positive."
        )

    max_token_id = (
        vocab_size - 1
    )

    candidate_dtypes = (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )

    for dtype in candidate_dtypes:

        if max_token_id <= np.iinfo(dtype).max:
            return np.dtype(
                dtype
            )

    raise ValueError(
        "Vocabulary is too large to represent "
        "with uint64 token IDs."
    )


def validate_token_ids(
    token_ids: Sequence[int],
    vocab_size: int,
) -> None:


    if len(token_ids) == 0:
        return

    min_id = min(
        token_ids
    )

    max_id = max(
        token_ids
    )

    if min_id < 0:
        raise ValueError(
            f"Invalid negative token ID: {min_id}"
        )

    if max_id >= vocab_size:
        raise ValueError(
            f"Token ID {max_id} is outside vocabulary "
            f"range [0, {vocab_size - 1}]."
        )


def encode_story(
    tokenizer: Tokenizer,
    story: str,
    vocab_size: int,
) -> list[int]:
    

    text = (
        story
        + EOT_TOKEN
    )

    ids = tokenizer.encode(
        text
    )

    validate_token_ids(
        ids,
        vocab_size,
    )

    return ids


def count_tokens(
    stories: Sequence[str],
    tokenizer: Tokenizer,
    vocab_size: int,
) -> int:


    total_tokens = 0

    for story in stories:

        ids = encode_story(
            tokenizer,
            story,
            vocab_size,
        )

        total_tokens += len(
            ids
        )

    return total_tokens


def write_tokenized_split(
    stories: Sequence[str],
    tokenizer: Tokenizer,
    vocab_size: int,
    output_path: Path,
    dtype: np.dtype,
    total_tokens: int,
) -> int:


    token_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(total_tokens,),
    )

    write_pos = 0

    for story in stories:


        ids = encode_story(
            tokenizer,
            story,
            vocab_size,
        )


        validate_token_ids(
            ids,
            vocab_size,
        )


        next_write_pos = (
            write_pos
            + len(ids)
        )


        token_array[
            write_pos:next_write_pos
        ] = np.asarray(
            ids,
            dtype=dtype,
        )


        write_pos = (
            next_write_pos
        )


    if write_pos != total_tokens:
        raise RuntimeError(
            "Token count mismatch: "
            f"expected {total_tokens}, "
            f"wrote {write_pos}."
        )


    token_array.flush()

    return write_pos


def build_stats(
    *,
    config: Config,
    vocab_size: int,
    dtype: np.dtype,
    train_story_count: int,
    valid_story_count: int,
    train_token_count: int,
    valid_token_count: int,
) -> dict:
    return {
        "input_path": str(config.input_path),
        "vocab_size": vocab_size,
        "special_token": EOT_TOKEN,
        "valid_ratio": config.valid_ratio,
        "seed": config.seed,
        "dtype": str(dtype),

        "train": {
            "stories": train_story_count,
            "tokens": train_token_count,
        },

        "valid": {
            "stories": valid_story_count,
            "tokens": valid_token_count,
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = parse_args()

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    tokenizer = load_tokenizer(
        config
    )

    vocab_size = get_vocab_size(
        tokenizer
    )

    logging.info(
        "Vocabulary size: %d",
        vocab_size,
    )


    stories = load_stories(
        config.input_path
    )

    logging.info(
        "Total stories: %d",
        len(stories),
    )


    train_stories, valid_stories = split_stories(
        stories,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
    )

    logging.info(
        "Train stories: %d",
        len(train_stories),
    )

    logging.info(
        "Valid stories: %d",
        len(valid_stories),
    )


    dtype = choose_token_dtype(
        vocab_size
    )

    logging.info(
        "Token dtype: %s",
        dtype,
    )


    train_token_count = count_tokens(
        train_stories,
        tokenizer,
        vocab_size,
    )

    valid_token_count = count_tokens(
        valid_stories,
        tokenizer,
        vocab_size,
    )

    logging.info(
        "Train tokens: %d",
        train_token_count,
    )

    logging.info(
        "Valid tokens: %d",
        valid_token_count,
    )


    train_path = (
        config.output_dir
        / "train.npy"
    )

    valid_path = (
        config.output_dir
        / "valid.npy"
    )

    write_tokenized_split(
        train_stories,
        tokenizer,
        vocab_size,
        train_path,
        dtype,
        train_token_count,
    )

    write_tokenized_split(
        valid_stories,
        tokenizer,
        vocab_size,
        valid_path,
        dtype,
        valid_token_count,
    )


    train = np.load(
        train_path,
        mmap_mode="r",
    )

    valid = np.load(
        valid_path,
        mmap_mode="r",
    )


    if train.ndim != 1:
        raise RuntimeError(
            f"Expected train.npy to be 1D, "
            f"got shape {train.shape}."
        )

    if valid.ndim != 1:
        raise RuntimeError(
            f"Expected valid.npy to be 1D, "
            f"got shape {valid.shape}."
        )


    if train.dtype != dtype:
        raise RuntimeError(
            f"Unexpected train dtype: "
            f"{train.dtype}, expected {dtype}."
        )

    if valid.dtype != dtype:
        raise RuntimeError(
            f"Unexpected valid dtype: "
            f"{valid.dtype}, expected {dtype}."
        )


    if len(train) != train_token_count:
        raise RuntimeError(
            "train.npy token count does not match "
            "the first-pass count."
        )

    if len(valid) != valid_token_count:
        raise RuntimeError(
            "valid.npy token count does not match "
            "the first-pass count."
        )


    if len(train) > 0:

        train_min = int(
            train.min()
        )

        train_max = int(
            train.max()
        )

        if train_min < 0:
            raise RuntimeError(
                f"Invalid train token ID: {train_min}"
            )

        if train_max >= vocab_size:
            raise RuntimeError(
                f"Invalid train token ID: {train_max}"
            )

    if len(valid) > 0:

        valid_min = int(
            valid.min()
        )

        valid_max = int(
            valid.max()
        )

        if valid_min < 0:
            raise RuntimeError(
                f"Invalid valid token ID: {valid_min}"
            )

        if valid_max >= vocab_size:
            raise RuntimeError(
                f"Invalid valid token ID: {valid_max}"
            )


    stats = build_stats(
        config=config,
        vocab_size=vocab_size,
        dtype=dtype,
        train_story_count=len(train_stories),
        valid_story_count=len(valid_stories),
        train_token_count=train_token_count,
        valid_token_count=valid_token_count,
    )

    stats_path = (
        config.output_dir
        / "preprocess_stats.json"
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

    logging.info(
        "Saved train data: %s",
        train_path,
    )

    logging.info(
        "Saved valid data: %s",
        valid_path,
    )

    logging.info(
        "Saved stats: %s",
        stats_path,
    )


if __name__ == "__main__":
    main()