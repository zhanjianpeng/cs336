from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


# =========================================================
# 1. 基本配置
# =========================================================

SPECIAL_TOKEN = "<|endoftext|>"

DEFAULT_VOCAB_SIZE = 10_000


# 用于验证 tokenizer 的几个例子
VALIDATION_TEXTS = {
    "english": (
        "Once upon a time, there was a little girl "
        "who loved reading stories."
    ),

    "chinese": (
        "从前有一个小女孩，她很喜欢读故事。"
    ),

    "special_token": (
        "This is the first story."
        "<|endoftext|>"
        "This is the second story."
    ),
}


# =========================================================
# 2. 命令行参数
# =========================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train and validate a Byte-level BPE tokenizer "
            "for TinyStories."
        )
    )

    # -----------------------------------------------------
    # 必须提供：
    # TinyStories 训练文件路径
    # -----------------------------------------------------

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the tokenizer training text file.",
    )

    # -----------------------------------------------------
    # 默认 10,000
    # -----------------------------------------------------

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=(
            "Target vocabulary size. "
            "Default: 10000."
        ),
    )

    # -----------------------------------------------------
    # 输出目录
    # -----------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/tinystories_tokenizer",
        help=(
            "Directory for vocab, merges, "
            "and experiment records."
        ),
    )

    # -----------------------------------------------------
    # 是否使用数据子集
    #
    # 例如调试阶段可以：
    #
    # python train_tokenizer.py ...
    #     --subset
    # -----------------------------------------------------

    parser.add_argument(
        "--subset",
        action="store_true",
        help=(
            "Mark this experiment as using "
            "a subset of TinyStories."
        ),
    )

    # -----------------------------------------------------
    # 可选备注
    # -----------------------------------------------------

    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes for the experiment record.",
    )

    return parser.parse_args()


# =========================================================
# 3. 获取训练数据基本信息
# =========================================================

def get_file_information(
    input_path: Path,
) -> dict[str, Any]:

    file_size_bytes = input_path.stat().st_size

    file_size_mib = (
        file_size_bytes
        / (1024 ** 2)
    )

    return {
        "path": str(input_path.resolve()),
        "filename": input_path.name,
        "size_bytes": file_size_bytes,
        "size_mib": round(
            file_size_mib,
            2,
        ),
    }


# =========================================================
# 4. 对若干文本执行 encode/decode
# =========================================================

def run_encoding_examples(
    tokenizer: Tokenizer,
) -> dict[str, dict[str, Any]]:

    results: dict[
        str,
        dict[str, Any],
    ] = {}

    for name, text in VALIDATION_TEXTS.items():

        # -------------------------------------------------
        # encode
        # -------------------------------------------------

        token_ids = tokenizer.encode(
            text
        )

        # -------------------------------------------------
        # decode
        # -------------------------------------------------

        decoded_text = tokenizer.decode(
            token_ids
        )

        # -------------------------------------------------
        # 记录实验结果
        # -------------------------------------------------

        results[name] = {
            "original_text": text,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
            "decoded_text": decoded_text,
            "roundtrip_equal": (
                decoded_text == text
            ),
        }

    return results


# =========================================================
# 5. 打印 encode/decode 示例
# =========================================================

def print_encoding_examples(
    examples: dict[str, dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("Encode / Decode Examples")
    print("=" * 70)

    for name, result in examples.items():

        print()
        print(f"[{name}]")

        print(
            "Original:"
        )
        print(
            result["original_text"]
        )

        print()
        print(
            "Token IDs:"
        )
        print(
            result["token_ids"]
        )

        print()
        print(
            f"Number of tokens: "
            f"{result['num_tokens']}"
        )

        print()
        print(
            "Decoded:"
        )
        print(
            result["decoded_text"]
        )

        print()
        print(
            "Round-trip equal:",
            result["roundtrip_equal"],
        )

        print("-" * 70)


# =========================================================
# 6. 验证保存前后的 tokenizer 是否一致
# =========================================================

def validate_saved_tokenizer(
    original_tokenizer: Tokenizer,
    loaded_tokenizer: Tokenizer,
    original_vocab: dict[int, bytes],
    original_merges: list[
        tuple[bytes, bytes]
    ],
) -> dict[str, bool]:

    # -----------------------------------------------------
    # 1. vocab 是否完全一致
    # -----------------------------------------------------

    vocab_equal = (
        original_tokenizer.vocab
        == loaded_tokenizer.vocab
        == original_vocab
    )

    # -----------------------------------------------------
    # 2. merge rules 是否完全一致
    # -----------------------------------------------------

    merges_equal = (
        original_tokenizer.merges
        == loaded_tokenizer.merges
        == original_merges
    )

    # -----------------------------------------------------
    # 3. 同一个普通文本 encode 是否一致
    # -----------------------------------------------------

    regular_text = (
        "Once upon a time, "
        "a little cat walked into the garden."
    )

    original_regular_ids = (
        original_tokenizer.encode(
            regular_text
        )
    )

    loaded_regular_ids = (
        loaded_tokenizer.encode(
            regular_text
        )
    )

    regular_encoding_equal = (
        original_regular_ids
        == loaded_regular_ids
    )

    # -----------------------------------------------------
    # 4. 含 special token 的文本
    # -----------------------------------------------------

    special_text = (
        "Story one."
        "<|endoftext|>"
        "Story two."
    )

    original_special_ids = (
        original_tokenizer.encode(
            special_text
        )
    )

    loaded_special_ids = (
        loaded_tokenizer.encode(
            special_text
        )
    )

    special_encoding_equal = (
        original_special_ids
        == loaded_special_ids
    )

    # -----------------------------------------------------
    # 5. decode 是否一致
    # -----------------------------------------------------

    decoding_equal = (
        original_tokenizer.decode(
            original_regular_ids
        )
        ==
        loaded_tokenizer.decode(
            loaded_regular_ids
        )
    )

    return {
        "vocab_equal": vocab_equal,
        "merges_equal": merges_equal,
        "regular_encoding_equal": (
            regular_encoding_equal
        ),
        "special_encoding_equal": (
            special_encoding_equal
        ),
        "decoding_equal": decoding_equal,
        "all_checks_passed": all(
            [
                vocab_equal,
                merges_equal,
                regular_encoding_equal,
                special_encoding_equal,
                decoding_equal,
            ]
        ),
    }


# =========================================================
# 7. 验证 special token
# =========================================================

def validate_special_token(
    tokenizer: Tokenizer,
) -> dict[str, Any]:

    ids = tokenizer.encode(
        SPECIAL_TOKEN
    )

    decoded = tokenizer.decode(
        ids
    )

    # special token 应该只对应一个 token ID
    is_single_token = (
        len(ids) == 1
    )

    roundtrip_equal = (
        decoded == SPECIAL_TOKEN
    )

    return {
        "special_token": SPECIAL_TOKEN,
        "token_ids": ids,
        "is_single_token": is_single_token,
        "decoded": decoded,
        "roundtrip_equal": roundtrip_equal,
        "passed": (
            is_single_token
            and roundtrip_equal
        ),
    }


# =========================================================
# 8. 找最长 vocabulary token
#
# 虽然不是你现在强制要求的记录项，
# 但原 CS336 实验要求观察最长 token，
# 所以顺手记录下来很有价值。
# =========================================================

def find_longest_vocab_token(
    vocab: dict[int, bytes],
) -> dict[str, Any]:

    token_id, token_bytes = max(
        vocab.items(),
        key=lambda item: len(
            item[1]
        ),
    )

    return {
        "token_id": token_id,
        "length_bytes": len(
            token_bytes
        ),
        "bytes_hex": (
            token_bytes.hex()
        ),
        "decoded_text": (
            token_bytes.decode(
                "utf-8",
                errors="replace",
            )
        ),
    }


# =========================================================
# 9. 保存实验记录 JSON
# =========================================================

def save_experiment_json(
    output_path: Path,
    record: dict[str, Any],
) -> None:

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            record,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# 10. 保存一个更适合阅读的 Markdown 记录
# =========================================================

def save_experiment_markdown(
    output_path: Path,
    record: dict[str, Any],
) -> None:

    data = record["dataset"]
    training = record["training"]
    validation = record["serialization_validation"]
    special = record["special_token_validation"]
    longest = record["longest_token"]
    examples = record["encoding_examples"]

    lines: list[str] = []

    lines.append(
        "# TinyStories Tokenizer Experiment"
    )

    lines.append("")

    lines.append(
        "## Dataset"
    )

    lines.append("")

    lines.append(
        f"- Source: `{data['path']}`"
    )

    lines.append(
        f"- File size: "
        f"{data['size_mib']} MiB "
        f"({data['size_bytes']} bytes)"
    )

    lines.append(
        f"- Using subset: "
        f"{record['used_subset']}"
    )

    lines.append("")

    lines.append(
        "## Training"
    )

    lines.append("")

    lines.append(
        f"- Target vocabulary size: "
        f"{training['target_vocab_size']}"
    )

    lines.append(
        f"- Actual vocabulary size: "
        f"{training['actual_vocab_size']}"
    )

    lines.append(
        f"- Number of merges: "
        f"{training['merge_count']}"
    )

    lines.append(
        f"- Training time: "
        f"{training['elapsed_seconds']:.3f} seconds"
    )

    lines.append(
        f"- Special token: "
        f"`{SPECIAL_TOKEN}`"
    )

    lines.append("")

    lines.append(
        "## Serialization Validation"
    )

    lines.append("")

    for key, value in validation.items():

        lines.append(
            f"- {key}: {value}"
        )

    lines.append("")

    lines.append(
        "## Special Token Validation"
    )

    lines.append("")

    lines.append(
        f"- Token IDs: "
        f"{special['token_ids']}"
    )

    lines.append(
        f"- Is single token: "
        f"{special['is_single_token']}"
    )

    lines.append(
        f"- Round-trip equal: "
        f"{special['roundtrip_equal']}"
    )

    lines.append("")

    lines.append(
        "## Longest Vocabulary Token"
    )

    lines.append("")

    lines.append(
        f"- Token ID: "
        f"{longest['token_id']}"
    )

    lines.append(
        f"- Length: "
        f"{longest['length_bytes']} bytes"
    )

    lines.append(
        f"- Decoded: "
        f"`{longest['decoded_text']}`"
    )

    lines.append("")

    lines.append(
        "## Encoding Examples"
    )

    for name, example in examples.items():

        lines.append("")

        lines.append(
            f"### {name}"
        )

        lines.append("")

        lines.append(
            f"Original:"
        )

        lines.append("")

        lines.append(
            example["original_text"]
        )

        lines.append("")

        lines.append(
            "Token IDs:"
        )

        lines.append("")

        lines.append(
            f"`{example['token_ids']}`"
        )

        lines.append("")

        lines.append(
            f"Decoded:"
        )

        lines.append("")

        lines.append(
            example["decoded_text"]
        )

        lines.append("")

        lines.append(
            f"Round-trip equal: "
            f"{example['roundtrip_equal']}"
        )

    if record["notes"]:

        lines.append("")

        lines.append(
            "## Notes"
        )

        lines.append("")

        lines.append(
            record["notes"]
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =========================================================
# 11. 主程序
# =========================================================

def main() -> None:

    args = parse_args()

    # -----------------------------------------------------
    # 1. 路径准备
    # -----------------------------------------------------

    input_path = Path(
        args.input_path
    )

    output_dir = Path(
        args.output_dir
    )

    # 检查训练文件
    if not input_path.exists():

        raise FileNotFoundError(
            f"Training file does not exist: "
            f"{input_path}"
        )

    if not input_path.is_file():

        raise ValueError(
            f"Input path is not a file: "
            f"{input_path}"
        )

    # Byte-level BPE 至少需要：
    #
    # 256 byte tokens
    # +
    # 1 special token
    #
    minimum_vocab_size = 257

    if args.vocab_size < minimum_vocab_size:

        raise ValueError(
            f"vocab_size must be at least "
            f"{minimum_vocab_size}, "
            f"but got {args.vocab_size}."
        )

    # 创建输出目录
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 2. 输出文件
    # -----------------------------------------------------

    vocab_path = (
        output_dir
        / "vocab.json"
    )

    merges_path = (
        output_dir
        / "merges.json"
    )

    experiment_json_path = (
        output_dir
        / "experiment.json"
    )

    experiment_markdown_path = (
        output_dir
        / "experiment.md"
    )

    # -----------------------------------------------------
    # 3. 获取数据基本信息
    # -----------------------------------------------------

    dataset_info = (
        get_file_information(
            input_path
        )
    )

    print()
    print("=" * 70)
    print("TinyStories BPE Tokenizer Training")
    print("=" * 70)

    print(
        f"Input file: "
        f"{dataset_info['path']}"
    )

    print(
        f"Dataset size: "
        f"{dataset_info['size_mib']} MiB"
    )

    print(
        f"Target vocab size: "
        f"{args.vocab_size}"
    )

    print(
        f"Special token: "
        f"{SPECIAL_TOKEN}"
    )

    print(
        f"Using subset: "
        f"{args.subset}"
    )

    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )

    print()

    # =====================================================
    # 4. 正式训练 BPE
    # =====================================================

    print(
        "Training tokenizer..."
    )

    start_time = (
        time.perf_counter()
    )

    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=args.vocab_size,
        special_tokens=[
            SPECIAL_TOKEN
        ],
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        "Training finished."
    )

    print(
        f"Elapsed time: "
        f"{elapsed_seconds:.3f} seconds"
    )

    print(
        f"Actual vocab size: "
        f"{len(vocab)}"
    )

    print(
        f"Number of merges: "
        f"{len(merges)}"
    )

    # =====================================================
    # 5. 创建 Tokenizer
    # =====================================================

    original_tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=[
            SPECIAL_TOKEN
        ],
    )

    # =====================================================
    # 6. 保存 vocab 和 merges
    # =====================================================

    print()
    print(
        "Saving tokenizer..."
    )

    original_tokenizer.save_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
    )

    print(
        f"Saved vocab to: "
        f"{vocab_path}"
    )

    print(
        f"Saved merges to: "
        f"{merges_path}"
    )

    # =====================================================
    # 7. 从磁盘重新加载
    # =====================================================

    print()
    print(
        "Reloading tokenizer from disk..."
    )

    loaded_tokenizer = (
        Tokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=[
                SPECIAL_TOKEN
            ],
        )
    )

    # =====================================================
    # 8. 验证保存 / 加载的一致性
    # =====================================================

    serialization_validation = (
        validate_saved_tokenizer(
            original_tokenizer=
                original_tokenizer,
            loaded_tokenizer=
                loaded_tokenizer,
            original_vocab=vocab,
            original_merges=merges,
        )
    )

    print()
    print("=" * 70)
    print("Serialization Validation")
    print("=" * 70)

    for (
        key,
        value,
    ) in serialization_validation.items():

        print(
            f"{key}: {value}"
        )

    # 如果保存前后不一致，
    # 这里直接报错，避免继续使用错误 tokenizer。
    if not serialization_validation[
        "all_checks_passed"
    ]:

        raise RuntimeError(
            "Tokenizer serialization validation failed."
        )

    # =====================================================
    # 9. 验证 special token
    # =====================================================

    special_token_validation = (
        validate_special_token(
            loaded_tokenizer
        )
    )

    print()
    print("=" * 70)
    print("Special Token Validation")
    print("=" * 70)

    print(
        "Encoded IDs:",
        special_token_validation[
            "token_ids"
        ],
    )

    print(
        "Single token:",
        special_token_validation[
            "is_single_token"
        ],
    )

    print(
        "Decoded:",
        special_token_validation[
            "decoded"
        ],
    )

    if not special_token_validation[
        "passed"
    ]:

        raise RuntimeError(
            "Special token validation failed."
        )

    # =====================================================
    # 10. 英文 / 中文 / special token 示例
    # =====================================================

    encoding_examples = (
        run_encoding_examples(
            loaded_tokenizer
        )
    )

    print_encoding_examples(
        encoding_examples
    )

    # 正常 UTF-8 文本应该可以完全 round-trip
    for (
        name,
        result,
    ) in encoding_examples.items():

        if not result[
            "roundtrip_equal"
        ]:

            raise RuntimeError(
                f"Round-trip test failed "
                f"for example: {name}"
            )

    # =====================================================
    # 11. 找最长 token
    # =====================================================

    longest_token = (
        find_longest_vocab_token(
            loaded_tokenizer.vocab
        )
    )

    print()
    print("=" * 70)
    print("Longest Vocabulary Token")
    print("=" * 70)

    print(
        "Token ID:",
        longest_token[
            "token_id"
        ],
    )

    print(
        "Length:",
        longest_token[
            "length_bytes"
        ],
        "bytes",
    )

    print(
        "Decoded:",
        repr(
            longest_token[
                "decoded_text"
            ]
        ),
    )

    # =====================================================
    # 12. 构造实验记录
    # =====================================================

    experiment_record: dict[
        str,
        Any,
    ] = {

        "experiment": (
            "TinyStories Byte-level "
            "BPE Tokenizer"
        ),

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "dataset": dataset_info,

        "used_subset": (
            args.subset
        ),

        "training": {
            "target_vocab_size":
                args.vocab_size,

            "actual_vocab_size":
                len(vocab),

            "merge_count":
                len(merges),

            "special_tokens": [
                SPECIAL_TOKEN
            ],

            "elapsed_seconds":
                elapsed_seconds,
        },

        "serialization_validation":
            serialization_validation,

        "special_token_validation":
            special_token_validation,

        "longest_token":
            longest_token,

        "encoding_examples":
            encoding_examples,

        "output_files": {
            "vocab": str(
                vocab_path.resolve()
            ),

            "merges": str(
                merges_path.resolve()
            ),

            "experiment_json": str(
                experiment_json_path.resolve()
            ),

            "experiment_markdown": str(
                experiment_markdown_path.resolve()
            ),
        },

        "environment": {
            "python_version":
                sys.version,

            "platform":
                platform.platform(),

            "cpu_count":
                os.cpu_count(),
        },

        "notes":
            args.notes,
    }

    # =====================================================
    # 13. 保存实验记录
    # =====================================================

    save_experiment_json(
        experiment_json_path,
        experiment_record,
    )

    save_experiment_markdown(
        experiment_markdown_path,
        experiment_record,
    )

    # =====================================================
    # 14. 最终结果
    # =====================================================

    print()
    print("=" * 70)
    print("All checks passed!")
    print("=" * 70)

    print()
    print(
        f"Vocabulary: "
        f"{vocab_path}"
    )

    print(
        f"Merges: "
        f"{merges_path}"
    )

    print(
        f"Experiment JSON: "
        f"{experiment_json_path}"
    )

    print(
        f"Experiment Markdown: "
        f"{experiment_markdown_path}"
    )


# =========================================================
# 15. Python script entry point
# =========================================================

if __name__ == "__main__":
    main()