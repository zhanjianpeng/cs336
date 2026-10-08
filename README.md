# CS336 Assignment 1: Basics

> 本仓库的本机配置使用 Conda 环境 `cs336`，请先阅读
> [环境配置与学习分工](./ENVIRONMENT.md)。下方保留官方说明；
> 本机运行测试请使用 `conda activate cs336` 后的 `python -m pytest`。

## 课程作业进度

Week 1 已完成：自行实现 byte-level BPE 训练、Tokenizer 编码和解码、特殊 token、
流式编码，以及 tokenizer 保存和加载。Week 2 的 Transformer 与 Week 3 的模型训练尚待完成。

官方骨架来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，
导入提交为 `a158843b20107949f1a8d7df1b05cd33b9166712`。
课程验收要求见 [课程作业 PDF](./Stanford%20CS336%20Assignment%201%20_%20Notion.pdf)。

### 目录

- `cs336_basics/bpe.py`：BPE vocabulary training。
- `cs336_basics/tokenizer.py`：Tokenizer 及保存、加载功能。
- `scripts/train_tokenizer.py`：训练和实验验证入口。
- `tests/`：官方测试、fixtures 和连接实现的 adapters。
- `reports/`：环境记录、测试结果和训练日志。
- `artifacts/tinystories_tokenizer/`：训练后的词表、合并规则和实验报告。
- `data/`：本地训练数据，不纳入 Git。

### 环境与测试

按 [ENVIRONMENT.md](./ENVIRONMENT.md) 创建并激活 Conda 环境 `cs336` 后，在项目根目录运行：

```bash
python -m pytest tests/test_train_bpe.py tests/test_tokenizer.py
```

Week 1 实测结果：**27 passed, 1 xfailed**。
`xfailed` 是官方预期失败的整段编码内存测试。
完整结果见 [reports/pytest-week1.txt](./reports/pytest-week1.txt)。
全套作业测试使用 `python -m pytest`，尚未实现的 Week 2 / Week 3 功能仍会失败。

### 复现 TinyStories tokenizer 实验

本次使用官方 `TinyStoriesV2-GPT4-train.txt` 的前缀子集，在达到 100 MiB 后遇到的
第一个完整 `<|endoftext|>` 行处结束，保留原始文档顺序。
实际大小为 **104,857,658 字节**，包含 **127,994 个完整故事**。
原始数据和子集不上传 Git；可以按下面的步骤重新准备：

```bash
mkdir -p data
curl -fL --retry 3 --retry-all-errors --continue-at - \
  -o data/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
LC_ALL=C awk -v target_bytes=104857600 '{ total_bytes += length($0) + 1; print; if (total_bytes >= target_bytes && $0 == "<|endoftext|>") exit }' \
  data/TinyStoriesV2-GPT4-train.txt > data/TinyStoriesV2-GPT4-train-prefix-100MiB.txt
sha256sum data/TinyStoriesV2-GPT4-train-prefix-100MiB.txt
```

本次子集的 SHA256：
`98d20772fd58201a12b5a021d4987b64e2bc360e3200bfc3591283e15dd11a00`。

训练并验证保存、加载和编码往返：

```bash
conda activate cs336
python scripts/train_tokenizer.py data/TinyStoriesV2-GPT4-train-prefix-100MiB.txt \
  --vocab-size 10000 \
  --output-dir artifacts/tinystories_tokenizer \
  --subset \
  --notes "TinyStoriesV2-GPT4 training prefix; approximately 100 MiB; complete document boundaries."
```

`--subset` 只标记实验使用子集，数据截取由上面的准备步骤完成。
脚本会保存 `vocab.json`、`merges.json`、`experiment.json` 和 `experiment.md`，
并检查加载前后词表、规则与编码结果一致，以及英文、中文和特殊 token 的往返转换。

本次结果：词表 **10,000**，merge 规则 **9,743**，训练耗时 **92.62 秒**，
峰值内存约 **655 MiB**，全部实验验证通过。
见 [实验报告](./artifacts/tinystories_tokenizer/experiment.md)、
[训练日志](./reports/tokenizer-training.log) 和
[资源记录](./reports/tokenizer-training-resources.txt)。

已有 tokenizer 可以通过 `Tokenizer.from_files` 加载；使用本次保存的两个 JSON 文件，
并传入特殊 token `<|endoftext|>`。模型训练、评估和生成命令将在 Week 3 实现后补充。

## 官方项目说明

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
