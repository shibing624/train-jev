# 更新日志

## [Unreleased]

#### fixes

- 仓库不再作为 Python 库打包：去掉 `pyproject.toml` 里的 setuptools 配置，不再支持 `pip install -e .` 以及 `.[serve]`、`.[dev]` 这些 extras。依赖只通过 `requirements.txt` 安装，在仓库根目录用 `python -m trainjev.train` 训练。
- README 方法图的虚线边写成了 `-.KL + 0.5·RPS.->`，`+` 不在引号里，GitHub 的 Mermaid 直接报 Lexical error，整张图渲染不出来。边标签改成 `-.->|"KL + 0.5 RPS"|`。
- 删掉 README / README_EN 里的 Project Structure 目录树。

## v0.1.0 (2026-10-14)

首次发布。

- 在 Qwen3.5 上复现 JEV System 1 配方：全部 Gated DeltaNet / attention / MLP 投影加 LoRA r16，24 槽位 fp32 决策头从 `lm_head` 初始化，损失为 KL + 0.5·RPS，30% 选项置换，按类型分层的 128 行 batch，多卡 DDP（`trainjev.train`）。
- 按问题类型做温度校准，并在语料上评测（`trainjev.evaluate`）。
- 兼容 TypeSafe System One 的服务，提供 `/v1/systemone`、`/v1/decide`、超过 16 个选项的锦标赛，以及 JEV-27B-VL 的代理模式（`trainjev.serve`）。
- Decision Index 视觉集上的零样本图像决策评测（`trainjev.eval_vision`）。
- Decision Index 0.3 辅助脚本：分层抽样和只含样本的 suite（`scripts/di_sample.py`），丢掉部分基准后重算榜单（`scripts/di_index.py`）。
- 13 个数据集的分类 PK：从 HF Hub 构建逐字节一致的 suite（`scripts/pk_build.py`），以及可断点续跑、带配对报告的评测（`scripts/pk_benchmark.py`）。
- Qwen3.8-27B 底座已端到端跑通；脚本 `scripts/run_train_27b.sh`。
- Python >= 3.12。
- Qwen3.5-4B 的结果见 [docs/results.md](docs/results.md)。
