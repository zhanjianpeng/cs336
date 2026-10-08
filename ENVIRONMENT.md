# 本机环境与学习分工

使用 Conda 环境 `cs336`（Python 3.12），不另建 `.venv`。
官方骨架来源：https://github.com/stanford-cs336/assignment1-basics
导入提交：`a158843b20107949f1a8d7df1b05cd33b9166712`。
当前仓库保留自己的 Git 历史与 origin。

## 重建环境

```bash
conda create -n cs336 python=3.12 pip -y
conda activate cs336
python -m pip install --no-cache-dir --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -e . uv --index-url https://pypi.org/simple
python -m pip check
```

PyTorch 使用官方 CUDA 12.6 wheel；版本符合项目的 `torch~=2.11.0` 要求。
安装方法参考：https://pytorch.org/get-started/previous-versions/#v2-11-0
其余依赖根据 `pyproject.toml` 安装，实际版本记录在 `reports/environment-freeze.txt`。
主包单独安装、依赖从 PyPI 补齐，是为了绕过本机访问 NVIDIA 包索引时的 SSL 连接中断。
两个安装步骤均须完成，不能只执行 `--no-deps` 那一步。
这不是对官方 `uv.lock` 的逐项复现：官方锁文件的 Linux CUDA 依赖与本机所选构建不同。

## 日常使用

在项目根目录运行：

```bash
conda activate cs336
which python
python --version
python -m pytest
```

如需使用作业中的 `uv run` 形式，明确使用当前 Conda 环境，并跳过环境同步：

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run --no-sync pytest
```

不要直接使用裸 `uv run pytest`，以免自动创建另一套 `.venv` 环境。
初始测试用于记录待实现功能；`NotImplementedError` 是骨架的预期状态。

## 分工

- AI 代劳：环境配置、官方骨架导入、依赖安装、初始测试及环境记录。
- 学生完成：BPE、模型、优化器及训练链路的核心实现，解释实现与实验现象。
- 一起完成：概念学习、测试结果分析、通过小例子定位 bug、审查学生代码。
- 学生执行 Git commit / push；数据、环境、checkpoint 不提交到 Git。

Week 1 已在约 100 MiB 的 TinyStories 完整文档子集上完成 tokenizer 训练。
数据准备与复现命令见 [README.md](./README.md)，实验结果见
[experiment.md](./artifacts/tinystories_tokenizer/experiment.md)。
