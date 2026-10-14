# 更新日志

## [Unreleased]

#### features

- 超过 16 个选项的 `choice` 题新增 `single` 和 `permute` 两种读法，和原有的 `tournament` 一起，对应 JEV-27B-VL `serve_decide.py` 的三种策略。服务端用 `--strategy` 选（默认 `single`，和 JEV-27B-VL 在 Qwen 底座上的默认一致），单个请求也可以在请求体里传 `"strategy"` 覆盖。`single` 一次前向读完：A–P 用训练过的决策头，Q–Z、AA、AB… 用底座 `lm_head` 的行、bias 为 0（`trainjev.modeling.choice_labels`）。`tournament` 现在把各组放在一个 batch 里打分。
- `trainjev.serve --device cpu`：没有 GPU 时用 fp32 在 CPU 上推理（很慢，只适合小规模核对）。
- 宽选项意图分类评测：`scripts/wide_build.py` 从 HF Hub 构建 CLINC150（150 类）/ BANKING77（77 类）/ MASSIVE en（59 类），后两个和 PK 题库逐字节一致；`scripts/run_wide.sh` 对三种策略各跑一遍并输出配对报告。`scripts/pk_benchmark.py run` 新增 `--strategy` 和 `--log_every`。

#### fixes

- README 里"`choice` 2–16 个选项，>16 个用锦标赛扩展到 256"的说法不准确：JEV-27B-VL 在 Qwen 底座上默认是一次前向读 256 个标签，锦标赛是它在 Gemma 底座上的默认做法。README / README_EN / docs 已改写。
- 仓库不再作为 Python 库打包：去掉 `pyproject.toml` 里的 setuptools 配置，不再支持 `pip install -e .` 以及 `.[serve]`、`.[dev]` 这些 extras。依赖只通过 `requirements.txt` 安装，在仓库根目录用 `python -m trainjev.train` 训练。
- README 方法图的虚线边写成了 `-.KL + 0.5·RPS.->`，`+` 不在引号里，GitHub 的 Mermaid 直接报 Lexical error，整张图渲染不出来。边标签改成 `-.->|"KL + 0.5 RPS"|`。
- 删掉 README / README_EN 里的 Project Structure 目录树。
- 推理服务只保留 TypeSafe System One API：去掉服务端的 `/v1/decide` 路由（JEV `serve_decide.py` 的原生接口，仓库里没有任何代码调用）。对外接口只有 `POST /v1/systemone` 和 `GET /health`。`--proxy` 模式和 `trainjev.evaluate` / `trainjev.eval_vision` 的 `--decide_url` 不变，它们作为客户端去调 JEV-27B-VL 自己的服务，用来做对比。

## v0.1.0 (2026-10-14)

首次发布。

- 在 Qwen3.5 上复现 JEV System 1 配方：全部 Gated DeltaNet / attention / MLP 投影加 LoRA r16，24 槽位 fp32 决策头从 `lm_head` 初始化，损失为 KL + 0.5·RPS，30% 选项置换，按类型分层的 128 行 batch，多卡 DDP（`trainjev.train`）。
- 按问题类型做温度校准，并在语料上评测（`trainjev.evaluate`）。
- 兼容 TypeSafe System One 的服务（`/v1/systemone`），支持超过 16 个选项的锦标赛，以及 JEV-27B-VL 的代理模式（`trainjev.serve`）。
- Decision Index 视觉集上的零样本图像决策评测（`trainjev.eval_vision`）。
- Decision Index 0.3 辅助脚本：分层抽样和只含样本的 suite（`scripts/di_sample.py`），丢掉部分基准后重算榜单（`scripts/di_index.py`）。
- 13 个数据集的分类 PK：从 HF Hub 构建逐字节一致的 suite（`scripts/pk_build.py`），以及可断点续跑、带配对报告的评测（`scripts/pk_benchmark.py`）。
- Qwen3.8-27B 底座已端到端跑通；脚本 `scripts/run_train_27b.sh`。
- Python >= 3.12。
- Qwen3.5-4B 的结果见 [docs/results.md](docs/results.md)。
