from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

PRETOKEN_PATTERN = re.compile(PAT)

BYTE_TOKENS = tuple(
    bytes([i])
    for i in range(256)
)

#去除重复special tokens
def _deduplicate_special_tokens(
    special_tokens: list[str],
) -> list[str]:

    result: list[str] = []
    seen: set[str] = set()

    for token in special_tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)

    return result

#合并pair
def _merge_one_pair(
    tokens: list[bytes],
    pair: tuple[bytes, bytes],
) -> list[bytes]:

    merged_token = pair[0] + pair[1]

    new_tokens: list[bytes] = []

    i = 0

    while i < len(tokens):

        if (
            i + 1 < len(tokens)
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            new_tokens.append(merged_token)
            i += 2

        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens

#Tokenizer class
class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):

        self.vocab = dict(vocab)
        self.merges = list(merges)

        if special_tokens is None:
            special_tokens = []

        self.special_tokens = _deduplicate_special_tokens(
            special_tokens
        )

        #建立 bytes -> token ID 映射
        self.bytes_to_id: dict[bytes, int] = {}

        for token_id, token_bytes in sorted(
            self.vocab.items()
        ):
            self.bytes_to_id.setdefault(
                token_bytes,
                token_id,
            )

        #加入不在词表中的 special token
        for token in self.special_tokens:

            token_bytes = token.encode("utf-8")

            if token_bytes not in self.bytes_to_id:
                new_id = max(
                    self.vocab.keys(),
                    default=-1,
                ) + 1
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

        self.special_to_id: dict[str, int] = {}

        for token in self.special_tokens:

            token_bytes = token.encode("utf-8")

            self.special_to_id[token] = (
                self.bytes_to_id[token_bytes]
            )

        #建立merge pair -> rank 映射
        self.merge_rank: dict[
            tuple[bytes, bytes],
            int,
        ] = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        #长token优先匹配
        if self.special_tokens:

            sorted_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            escaped_tokens = [
                re.escape(token)
                for token in sorted_special_tokens
            ]

            pattern = "|".join(
                escaped_tokens
            )

            self.special_pattern = re.compile(
                pattern
            )
        else:
            self.special_pattern = None

        self.max_special_token_length = max(
            (
                len(token)
                for token in self.special_tokens
            ),
            default=0,
        )

    #从文件加载tokenizer
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        这里定义一种简单、可逆的保存格式：

        vocab.json

        {
            "0": "00",
            "1": "01",
            ...
            "97": "61",
            ...
        }

        value 使用 bytes.hex() 保存。

        merges.json

        [
            ["74", "68"],
            ["20", "63"],
            ...
        ]

        同样使用 hexadecimal string 保存 bytes。
        """

        #加载 vocab
        with open(
            vocab_filepath,
            "r",
            encoding="utf-8",
        ) as f:

            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}

        for token_id, token_hex in raw_vocab.items():

            vocab[int(token_id)] = bytes.fromhex(
                token_hex
            )

        #加载 merges
        with open(
            merges_filepath,
            "r",
            encoding="utf-8",
        ) as f:

            raw_merges = json.load(f)

        merges: list[
            tuple[bytes, bytes]
        ] = []

        for left_hex, right_hex in raw_merges:

            merges.append(
                (
                    bytes.fromhex(left_hex),
                    bytes.fromhex(right_hex),
                )
            )

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    #保存tokenizer到文件
    def save_files(
        self,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
    ) -> None:

        #保存 vocab
        serializable_vocab = {
            str(token_id): token_bytes.hex()
            for token_id, token_bytes
            in self.vocab.items()
        }

        with open(
            vocab_filepath,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                serializable_vocab,
                f,
                indent=2,
            )

        #保存 merges
        serializable_merges = [
            [
                left.hex(),
                right.hex(),
            ]
            for left, right in self.merges
        ]

        with open(
            merges_filepath,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                serializable_merges,
                f,
                indent=2,
            )

    #对单个 pre-token 执行 BPE
    def _encode_pretoken(
        self,
        pretoken: str,
    ) -> list[int]:

        encoded = pretoken.encode("utf-8")

        tokens: list[bytes] = [
            BYTE_TOKENS[b]
            for b in encoded
        ]

        while len(tokens) >= 2:

            best_pair: tuple[
                bytes,
                bytes,
            ] | None = None

            best_rank: int | None = None

            for i in range(len(tokens) - 1):

                pair = (
                    tokens[i],
                    tokens[i + 1],
                )

                rank = self.merge_rank.get(
                    pair
                )

                if rank is None:
                    continue

                if (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = _merge_one_pair(
                tokens,
                best_pair,
            )

        token_ids: list[int] = []

        for token_bytes in tokens:

            token_id = self.bytes_to_id[
                token_bytes
            ]

            token_ids.append(
                token_id
            )

        return token_ids

    #编码普通文本（不包含 special token）
    def _encode_regular_text(
        self,
        text: str,
    ) -> Iterator[int]:

        for match in PRETOKEN_PATTERN.finditer(
            text
        ):

            pretoken = match.group()

            token_ids = self._encode_pretoken(
                pretoken
            )

            for token_id in token_ids:
                yield token_id

    #encode
    def encode(
        self,
        text: str,
    ) -> list[int]:

        result: list[int] = []

        # 没有 special tokens
        if self.special_pattern is None:

            result.extend(
                self._encode_regular_text(
                    text
                )
            )

            return result

        # 有 special tokens
        previous_end = 0

        for match in self.special_pattern.finditer(
            text
        ):
            normal_text = text[
                previous_end:match.start()
            ]

            result.extend(
                self._encode_regular_text(
                    normal_text
                )
            )

            special_token = match.group()

            result.append(
                self.special_to_id[
                    special_token
                ]
            )

            previous_end = match.end()

        remaining_text = text[
            previous_end:
        ]

        result.extend(
            self._encode_regular_text(
                remaining_text
            )
        )

        return result

    #decode
    def decode(
        self,
        ids: list[int],
    ) -> str:

        token_bytes_list: list[bytes] = []

        for token_id in ids:

            token_bytes = self.vocab[
                token_id
            ]

            token_bytes_list.append(
                token_bytes
            )

        combined_bytes = b"".join(
            token_bytes_list
        )

        text = combined_bytes.decode(
            "utf-8",
            errors="replace",
        )

        return text

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:

        pending = ""

        BUFFER_SIZE = 4096

        for piece in iterable:

            pending += piece

            while len(pending) >= BUFFER_SIZE:

                cut_position = (
                    self._find_safe_cut_position(
                        pending
                    )
                )

                if cut_position == 0:
                    break

                safe_text = pending[
                    :cut_position
                ]

                pending = pending[
                    cut_position:
                ]

                token_ids = self.encode(
                    safe_text
                )

                for token_id in token_ids:
                    yield token_id

        if pending:

            token_ids = self.encode(
                pending
            )

            for token_id in token_ids:
                yield token_id

    def _find_safe_cut_position(
        self,
        text: str,
    ) -> int:

        reserve = max(
            2,
            self.max_special_token_length,
        )

        safe_limit = len(text) - reserve

        if safe_limit <= 0:
            return 0

        best_cut = 0

        if self.special_pattern is None:

            for match in PRETOKEN_PATTERN.finditer(
                text
            ):

                if match.end() > safe_limit:
                    break

                matched_text = match.group()

                if (
                    matched_text
                    and not matched_text[-1].isspace()
                ):
                    best_cut = match.end()

            return best_cut

        previous_end = 0

        for special_match in (
            self.special_pattern.finditer(text)
        ):

            region_start = previous_end
            region_end = special_match.start()

            for match in PRETOKEN_PATTERN.finditer(
                text,
                region_start,
                region_end,
            ):

                if match.end() > safe_limit:
                    break

                matched_text = match.group()

                if (
                    matched_text
                    and not matched_text[-1].isspace()
                ):
                    best_cut = match.end()

            if special_match.end() <= safe_limit:

                best_cut = special_match.end()

            previous_end = special_match.end()

        for match in PRETOKEN_PATTERN.finditer(
            text,
            previous_end,
        ):

            if match.end() > safe_limit:
                break

            matched_text = match.group()

            if (
                matched_text
                and not matched_text[-1].isspace()
            ):
                best_cut = match.end()

        return best_cut