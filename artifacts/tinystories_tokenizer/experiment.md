# TinyStories Tokenizer Experiment

## Dataset

- Source: `/home/zjp2025/cs336/data/TinyStoriesV2-GPT4-train-prefix-100MiB.txt`
- File size: 100.0 MiB (104857658 bytes)
- Using subset: True

## Training

- Target vocabulary size: 10000
- Actual vocabulary size: 10000
- Number of merges: 9743
- Training time: 92.617 seconds
- Special token: `<|endoftext|>`

## Serialization Validation

- vocab_equal: True
- merges_equal: True
- regular_encoding_equal: True
- special_encoding_equal: True
- decoding_equal: True
- all_checks_passed: True

## Special Token Validation

- Token IDs: [256]
- Is single token: True
- Round-trip equal: True

## Longest Vocabulary Token

- Token ID: 9591
- Length: 17 bytes
- Decoded: ` enthusiastically`

## Encoding Examples

### english

Original:

Once upon a time, there was a little girl who loved reading stories.

Token IDs:

`[429, 438, 259, 398, 44, 401, 283, 259, 391, 475, 659, 502, 3833, 2099, 46]`

Decoded:

Once upon a time, there was a little girl who loved reading stories.

Round-trip equal: True

### chinese

Original:

从前有一个小女孩，她很喜欢读故事。

Token IDs:

`[228, 187, 142, 229, 137, 141, 230, 156, 137, 228, 184, 128, 228, 184, 170, 229, 176, 143, 229, 165, 179, 229, 173, 169, 239, 188, 140, 229, 165, 185, 229, 190, 136, 229, 150, 156, 230, 172, 162, 232, 175, 187, 230, 149, 133, 228, 186, 139, 227, 128, 130]`

Decoded:

从前有一个小女孩，她很喜欢读故事。

Round-trip equal: True

### special_token

Original:

This is the first story.<|endoftext|>This is the second story.

Token IDs:

`[1544, 433, 263, 1185, 1002, 46, 256, 1544, 433, 263, 4716, 1002, 46]`

Decoded:

This is the first story.<|endoftext|>This is the second story.

Round-trip equal: True

## Notes

Source: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt. Selection: file prefix through the first complete <|endoftext|> line at or after 104857600 bytes, preserving original document order. Actual size: 104857658 bytes; complete documents: 127994. SHA256: 98d20772fd58201a12b5a021d4987b64e2bc360e3200bfc3591283e15dd11a00. This experiment uses a training subset, not the full training file.