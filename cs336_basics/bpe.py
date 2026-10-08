import heapq
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 1. 词表初始化
def init_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:

    vocab: dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        next_id = len(vocab)
        vocab[next_id] = token_bytes

    return vocab

# 2. 词频统计
def count_pretokens(
    input_path: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:

    pretoken_counts: dict[tuple[bytes, ...], int] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        escaped_tokens = [
            re.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        ]
        special_pattern = "|".join(escaped_tokens)
        segments = re.split(
            special_pattern,
            text,
        )
    else:
        segments = [text]

    for segment in segments:
        for match in re.finditer(PAT, segment):
            pretoken = match.group()
            encoded = pretoken.encode("utf-8")
            byte_tuple = tuple(
                bytes([b])
                for b in encoded
            )
            if byte_tuple in pretoken_counts:
                pretoken_counts[byte_tuple] += 1
            else:
                pretoken_counts[byte_tuple] = 1
    return pretoken_counts


# 3. 统计pre-token内部的pair frequency
def local_pair_counts(
    tokens: tuple[bytes, ...],
) -> dict[tuple[bytes, bytes], int]:

    counts: dict[
        tuple[bytes, bytes],
        int,
    ] = {}

    for i in range(len(tokens) - 1):
        pair = (
            tokens[i],
            tokens[i + 1],
        )
        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
    return counts


# 一次性建立 pair_counts
# 同时建立 pair -> pretokens 的倒排索引
def build_pair_statistics(
    pretoken_counts: dict[
        tuple[bytes, ...],
        int,
    ],
) -> tuple[
    dict[tuple[bytes, bytes], int],
    dict[
        tuple[bytes, bytes],
        set[tuple[bytes, ...]],
    ],
]:

    # 全局 pair frequency
    pair_counts: dict[
        tuple[bytes, bytes],
        int,
    ] = {}

    # 倒排索引：
    pair_to_pretokens: dict[
        tuple[bytes, bytes],
        set[tuple[bytes, ...]],
    ] = {}

    for tokens, frequency in pretoken_counts.items():

        local_counts = local_pair_counts(tokens)

        for pair, occurrences in local_counts.items():
            weighted_count = (
                occurrences * frequency
            )
            if pair in pair_counts:
                pair_counts[pair] += weighted_count
            else:
                pair_counts[pair] = weighted_count

            if pair not in pair_to_pretokens:
                pair_to_pretokens[pair] = set()

            pair_to_pretokens[pair].add(tokens)

    return pair_counts, pair_to_pretokens

# 使用 heap 选择 best pair
class ReverseLexPair:
    __slots__ = ("pair",)
    def __init__(
        self,
        pair: tuple[bytes, bytes],
    ):
        self.pair = pair
    def __lt__(
        self,
        other: "ReverseLexPair",
    ) -> bool:
        return self.pair > other.pair

def build_pair_heap(
    pair_counts: dict[
        tuple[bytes, bytes],
        int,
    ],
) -> list[tuple[int, ReverseLexPair]]:

    heap: list[
        tuple[int, ReverseLexPair]
    ] = []

    for pair, frequency in pair_counts.items():
        heap.append(
            (
                -frequency,
                ReverseLexPair(pair),
            )
        )
    heapq.heapify(heap)
    return heap


def push_pair_to_heap(
    heap: list[
        tuple[int, ReverseLexPair]
    ],
    pair: tuple[bytes, bytes],
    frequency: int,
) -> None:

    if frequency <= 0:
        return

    heapq.heappush(
        heap,
        (
            -frequency,
            ReverseLexPair(pair),
        ),
    )


def select_pair_fast(
    pair_counts: dict[
        tuple[bytes, bytes],
        int,
    ],
    heap: list[
        tuple[int, ReverseLexPair]
    ],
) -> tuple[bytes, bytes] | None:

    while heap:
        negative_frequency, wrapper = heap[0]
        pair = wrapper.pair
        stored_frequency = -negative_frequency
        current_frequency = pair_counts.get(pair)

        if current_frequency == stored_frequency:
            return pair

        heapq.heappop(heap)

    return None

# 4. merge tokens
def merge_tokens(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:

    merged_token = (
        pair[0] + pair[1]
    )

    new_tokens: list[bytes] = []

    i = 0

    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            new_tokens.append(
                merged_token
            )
            i += 2
        else:
            new_tokens.append(
                tokens[i]
            )
            i += 1
    return tuple(new_tokens)

def merge_pair_incremental(
    pretoken_counts: dict[
        tuple[bytes, ...],
        int,
    ],
    pair_counts: dict[
        tuple[bytes, bytes],
        int,
    ],
    pair_to_pretokens: dict[
        tuple[bytes, bytes],
        set[tuple[bytes, ...]],
    ],
    pair_heap: list[
        tuple[int, ReverseLexPair]
    ],
    pair: tuple[bytes, bytes],
) -> None:

    # 1. 找到真正受到这次 merge 影响的 pre-token
    affected_pretokens = list(
        pair_to_pretokens.get(
            pair,
            (),
        )
    )

    if not affected_pretokens:
        return

    # 2. 新 pre-token 暂时放这里
    produced_counts: dict[
        tuple[bytes, ...],
        int,
    ] = {}

    pair_deltas: dict[
        tuple[bytes, bytes],
        int,
    ] = {}

    # 3. 删除旧 pre-token 对 pair statistics 的贡献
    for tokens in affected_pretokens:

        frequency = pretoken_counts.pop(
            tokens
        )

        old_local_counts = (
            local_pair_counts(tokens)
        )

        for old_pair, occurrences in old_local_counts.items():
            weighted_count = (
                occurrences * frequency
            )

            pair_deltas[old_pair] = (
                pair_deltas.get(
                    old_pair,
                    0,
                )
                - weighted_count
            )

            # 更新倒排索引
            bucket = pair_to_pretokens[
                old_pair
            ]

            bucket.discard(tokens)

            if not bucket:
                del pair_to_pretokens[
                    old_pair
                ]

        new_tokens = merge_tokens(
            tokens,
            pair,
        )

        if new_tokens in produced_counts:
            produced_counts[
                new_tokens
            ] += frequency
        else:
            produced_counts[
                new_tokens
            ] = frequency

    for (
        new_tokens,
        added_frequency,
    ) in produced_counts.items():

        if new_tokens in pretoken_counts:
            pretoken_counts[
                new_tokens
            ] += added_frequency
        else:
            pretoken_counts[
                new_tokens
            ] = added_frequency

        new_local_counts = (
            local_pair_counts(
                new_tokens
            )
        )

        for (
            new_pair,
            occurrences,
        ) in new_local_counts.items():
            weighted_count = (
                occurrences
                * added_frequency
            )
            pair_deltas[new_pair] = (
                pair_deltas.get(
                    new_pair,
                    0,
                )
                + weighted_count
            )

            if (
                new_pair
                not in pair_to_pretokens
            ):
                pair_to_pretokens[
                    new_pair
                ] = set()

            pair_to_pretokens[
                new_pair
            ].add(
                new_tokens
            )

    for (
        changed_pair,
        delta,
    ) in pair_deltas.items():

        new_count = (
            pair_counts.get(
                changed_pair,
                0,
            )
            + delta
        )

        if new_count <= 0:

            pair_counts.pop(
                changed_pair,
                None,
            )

        else:
            pair_counts[
                changed_pair
            ] = new_count

            push_pair_to_heap(
                pair_heap,
                changed_pair,
                new_count,
            )

def maybe_rebuild_heap(
    pair_counts: dict[
        tuple[bytes, bytes],
        int,
    ],
    heap: list[
        tuple[int, ReverseLexPair]
    ],
) -> list[
    tuple[int, ReverseLexPair]
]:

    threshold = max(
        10000,
        4 * len(pair_counts),
    )

    if len(heap) > threshold:
        return build_pair_heap(
            pair_counts
        )

    return heap

# 5. BPE Training
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:

    # 1. 初始化 vocabulary
    vocab = init_vocab(
        special_tokens
    )

    # 2. Pre-tokenization
    pretoken_counts = count_pretokens(
        input_path,
        special_tokens,
    )

    # pair_counts 完整计算一次
    # 创建倒排索引
    (
        pair_counts,
        pair_to_pretokens,
    ) = build_pair_statistics(
        pretoken_counts
    )

    # 构建 max-pair heap
    pair_heap = build_pair_heap(
        pair_counts
    )

    # 记录 merge rules
    merges: list[
        tuple[bytes, bytes]
    ] = []

    # BPE training loop
    while len(vocab) < vocab_size:

        best_pair = select_pair_fast(
            pair_counts,
            pair_heap,
        )

        if best_pair is None:
            break

        # 记录 merge
        merges.append(
            best_pair
        )

        # 新 vocabulary token
        new_token = (
            best_pair[0]
            + best_pair[1]
        )

        new_token_id = len(vocab)

        vocab[
            new_token_id
        ] = new_token

        # 只更新受 best_pair 影响的 pre-token
        merge_pair_incremental(
            pretoken_counts,
            pair_counts,
            pair_to_pretokens,
            pair_heap,
            best_pair,
        )

        pair_heap = maybe_rebuild_heap(
            pair_counts,
            pair_heap,
        )

    return vocab, merges