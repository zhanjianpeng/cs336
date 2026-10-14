# CS336 Assignment 1: Basics

> 本仓库的本机配置使用 Conda 环境 `cs336`，请先阅读
> [环境配置与学习分工](./ENVIRONMENT.md)。下方保留官方说明；
> 本机运行测试请使用 `conda activate cs336` 后的 `python -m pytest`。

## 课程作业进度

Week 1 已完成：自行实现 byte-level BPE 训练、Tokenizer 编码和解码、特殊 token、
流式编码，以及 tokenizer 保存和加载。
Week 2 已完成模型实现与测试：基础层、SwiGLU、RoPE、Softmax、缩放点积注意力、
多头因果自注意力、Transformer Block 和 Transformer LM。
Week 3 的训练基础组件、数据预处理、TinyStories 5,000 步基线训练与文本生成已完成，
模型与优化器恢复验证通过；个人实验总结与最终仓库提交仍需完成。

官方骨架来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，
导入提交为 `a158843b20107949f1a8d7df1b05cd33b9166712`。
课程验收要求见 [课程作业 PDF](./Stanford%20CS336%20Assignment%201%20_%20Notion.pdf)。

### 目录

- `cs336_basics/bpe.py`：BPE vocabulary training。
- `cs336_basics/tokenizer.py`：Tokenizer 及保存、加载功能。
- `cs336_basics/layers.py`：Linear、Embedding、RMSNorm、SiLU、SwiGLU 和 RoPE。
- `cs336_basics/nn_utils.py`：数值稳定的 Softmax、交叉熵和整体梯度裁剪。
- `cs336_basics/attention.py`：缩放点积注意力及支持可选 RoPE 的多头因果自注意力。
- `cs336_basics/transformer.py`：Transformer Block 和 Transformer LM。
- `cs336_basics/optim.py`：AdamW 及带预热的余弦学习率调度。
- `cs336_basics/data.py`：支持 NumPy 内存映射数组的随机批次采样。
- `cs336_basics/checkpoint.py`：模型、优化器状态和迭代次数的保存与恢复。
- `scripts/train_tokenizer.py`：训练和实验验证入口。
- `scripts/prepare_data.py`：按完整故事划分训练、验证集，并写出 token 数组。
- `scripts/train.py`：可配置的训练、验证及 checkpoint 恢复入口。
- `scripts/generate.py`：模型加载、温度与 top-p 采样及自回归文本生成。
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
全套作业单元测试使用 `python -m pytest`；模型训练与文本生成实验单独验收。
当前全套官方测试结果为 **47 passed, 1 xfailed in 42.87s**，
见 [最终测试日志](./reports/pytest-required-final.txt)。

### Week 2 模型验收

在 Conda 环境 `cs336` 中运行全部模型测试和 Softmax 测试：

```bash
python -m pytest tests/test_model.py tests/test_nn_utils.py::test_softmax_matches_pytorch -v --tb=short
```

本次结果：**14 passed in 0.35s**。
完整日志见 [reports/pytest-week2.txt](./reports/pytest-week2.txt)。
测试覆盖基础层、三维和四维缩放点积注意力、无 RoPE 和带 RoPE 的多头注意力、
Transformer Block，以及完整输入和截短输入的 Transformer LM。

主要模块的输入和输出形状：

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| Linear | `(..., in_features)` | `(..., out_features)` |
| Embedding | `(...)` 的整数 token ID | `(..., embedding_dim)` |
| Transformer Block | `(..., seq_len, d_model)` | `(..., seq_len, d_model)` |
| Transformer LM | `(batch_size, seq_len)` 的整数 token ID | `(batch_size, seq_len, vocab_size)` 的未归一化 logits |

Transformer LM 支持 `seq_len <= context_length`。
本地模型的构造参数 `theta` 对应官方 adapter 的 `rope_theta`；
`tests/adapters.py` 负责官方接口与本地模块参数名称的映射。

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
并传入特殊 token `<|endoftext|>`。模型训练、验证与生成命令见下文。

### Week 3 训练基础组件

交叉熵、AdamW、学习率调度、梯度裁剪、批次采样和 checkpoint 已逐项通过官方测试。
另外检查了交叉熵梯度、只读内存映射输入、无梯度参数，以及 checkpoint 恢复后继续更新的一致性。

统一运行这些组件的官方测试：

```bash
python -m pytest tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py -v --tb=short
```

逐项验收记录：

- [交叉熵](./reports/pytest-week3-cross-entropy.txt)。
- [AdamW](./reports/pytest-week3-adamw.txt)。
- [学习率调度](./reports/pytest-week3-lr-schedule.txt)。
- [梯度裁剪](./reports/pytest-week3-gradient-clipping.txt)。
- [批次采样](./reports/pytest-week3-data-loading.txt)。
- [Checkpoint](./reports/pytest-week3-checkpoint.txt)。

### 准备 TinyStories 训练与验证数据

使用上文的 100 MiB 前缀子集和已保存的 tokenizer，按完整故事随机划分：
90% 用于模型训练，10% 用作该子集的内部验证集。随机种子为 42。
每个故事保留 `<|endoftext|>`，训练和验证使用相同的词表与合并规则。

```bash
conda activate cs336
python -m scripts.prepare_data \
  --input data/TinyStoriesV2-GPT4-train-prefix-100MiB.txt \
  --vocab artifacts/tinystories_tokenizer/vocab.json \
  --merges artifacts/tinystories_tokenizer/merges.json \
  --output-dir data/tinystories \
  --valid-ratio 0.1 \
  --seed 42
```

输出文件为 `data/tinystories/train.npy`、`valid.npy` 和 `preprocess_stats.json`。
token 数组为一维 `uint16`；训练入口应通过 `np.load(..., mmap_mode="r")` 读取，
再由 `get_batch` 转为所需设备上的 `torch.long` 批次。
预处理脚本先逐篇编码计数，再逐篇编码并写入内存映射数组，因此会编码两遍。

小规模验证见 [reports/preprocess-smoke.txt](./reports/preprocess-smoke.txt)，
覆盖完整故事划分、两部分互不重叠、英文与中文往返、结束标记、ID 范围，
以及相同种子的输出可复现性。

本次正式输出：

| 划分 | 故事数 | Token 数 |
| --- | ---: | ---: |
| 训练 | 115,195 | 22,914,183 |
| 验证 | 12,799 | 2,555,892 |

预处理耗时 **9 分 38.79 秒**，峰值内存约 **674 MiB**。
两份数组均通过形状、类型、ID 范围、结束标记数量、源故事首尾抽样及 `get_batch` 检查。
原始语料、词表、合并规则、脚本及输出数组的 SHA256 保存在验证报告中。

- [处理日志](./reports/preprocess-full.txt)。
- [资源记录](./reports/preprocess-resources.txt)。
- [统计副本](./reports/preprocess-stats.json)。
- [输出验证](./reports/preprocess-validation.txt) 与 [完整验证报告](./reports/preprocess-validation.json)。

### TinyStories 模型训练与验证

以下命令使用同一个 100 MiB 子集，在 `cuda:0` 上训练 5,000 步，共处理 40,960,000 tokens。
模型为 4 层、隐藏维度 512、16 个注意力头，使用 float32；每 250 步估计验证损失。

```bash
conda activate cs336
python -u -m scripts.train \
  --train-data data/tinystories/train.npy \
  --valid-data data/tinystories/valid.npy \
  --output-dir checkpoints/tinystories-baseline \
  --vocab-size 10000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size 32 --max-steps 5000 \
  --max-lr 0.001 --min-lr 0.0001 --warmup-steps 200 --cosine-cycle-steps 5000 \
  --beta1 0.9 --beta2 0.95 --eps 1e-8 --weight-decay 0.1 --max-grad-norm 1.0 \
  --log-interval 50 --eval-interval 250 --eval-batches 10 --checkpoint-interval 2500 \
  --device cuda:0 --seed 42 > reports/tinystories-training.txt 2>&1
```

脚本保存 `train_config.json`、定期 checkpoint 和 `checkpoint_final.pt`。
训练与验证损失以 JSON 行写入日志。验证使用 `eval` 和 `no_grad`，并在结束后恢复训练模式。

`--max-steps` 表示总训练步数上限。恢复时，checkpoint 已完成步数超过该上限会明确报错；
等于上限时不再更新参数。相关检查见 [恢复边界验证](./reports/train-resume-boundary-review.txt)。

恢复训练时，使用同一组数据与模型参数，添加
`--resume-from checkpoints/tinystories-baseline/checkpoint_final.pt`，
并将 `--max-steps` 设为更大的总步数。脚本从已保存的迭代位置继续调度学习率。
当前 checkpoint 保存模型、优化器和迭代次数，不保存随机数状态，因此恢复训练不会复现
不中断训练时的同一批次序列。

本次基线模型有 **22,696,448 个参数**，训练耗时 **585.17 秒**。
首次记录的 50 步平均训练损失为 **7.7732**，最后 50 步为 **1.6164**；
最终内部验证损失为 **1.7257**，困惑度约 **5.6167**。
加载最终 checkpoint 后继续到第 5,001 步，模型参数发生更新，AdamW 步数与迭代次数一致。

![TinyStories 训练与验证损失曲线](./reports/tinystories-loss-curve.png)

- [实验记录](./reports/tinystories-experiment.md)。
- [训练配置](./reports/tinystories-training-config.json) 与 [指标 CSV](./reports/tinystories-training-metrics.csv)。
- [训练日志](./reports/tinystories-training.txt) 与 [结果汇总](./reports/tinystories-training-summary.json)。
- [恢复训练日志](./reports/tinystories-resume-validation.txt) 与 [状态核对](./reports/tinystories-results-verification.txt)。
- [曲线 PDF](./reports/tinystories-loss-curve.pdf)。

### 评估已训练的 checkpoint

恢复最终 checkpoint，并把总步数上限设为已完成的 5,000 步，此时不执行额外参数更新。
以下命令只计算最终验证损失，并将配置及 checkpoint 保存到单独的评估目录：

```bash
python -u -m scripts.train \
  --train-data data/tinystories/train.npy --valid-data data/tinystories/valid.npy \
  --output-dir checkpoints/tinystories-evaluation \
  --vocab-size 10000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size 32 --max-steps 5000 \
  --max-lr 0.001 --min-lr 0.0001 --warmup-steps 200 --cosine-cycle-steps 5000 \
  --beta1 0.9 --beta2 0.95 --eps 1e-8 --weight-decay 0.1 --max-grad-norm 1.0 \
  --eval-batches 10 --device cuda:0 --seed 42 \
  --resume-from checkpoints/tinystories-baseline/checkpoint_final.pt
```

此命令已验证不会更新模型与优化器状态，见 [评估日志](./reports/tinystories-evaluation.txt)
和 [状态核对](./reports/tinystories-evaluation-validation.txt)。验证批次重新随机采样，
因此损失可与训练结束时的采样估计略有不同。

### 文本生成

在项目根目录、Conda 环境 `cs336` 中运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u -m scripts.generate \
  --config reports/tinystories-training-config.json \
  --checkpoint checkpoints/tinystories-baseline/checkpoint_final.pt \
  --vocab artifacts/tinystories_tokenizer/vocab.json \
  --merges artifacts/tinystories_tokenizer/merges.json \
  --prompt "Once upon a time" \
  --max-new-tokens 192 --temperature 0.8 --top-p 0.9 --device cpu --seed 42
```

脚本保留完整输出历史，每次只把最近的模型上下文窗口送入 Transformer；
遇到 `<|endoftext|>` 或达到新增 token 上限时停止。推理使用 `eval` 和 `no_grad`。

实际验证中，两段故事分别生成 138 和 97 个新 token，并正常结束。
300-token prompt、零新增 token、同种子可复现性均通过检查。
采样检查覆盖精确 top-p 阈值与 `top_p=1` 的小概率 token。

- [生成示例](./reports/generation-examples.md) 与 [结构化记录](./reports/generation-examples.json)。
- [采样与控制流程检查](./reports/generation-validation.txt)。
- [真实模型验证](./reports/generation-examples-validation.txt) 与 [可复现性记录](./reports/generation-repeatability.txt)。

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
