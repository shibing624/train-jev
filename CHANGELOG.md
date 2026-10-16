# 更新日志

## [Unreleased]

#### fixes

- README 对照表里的官方模型名称不准确：`JEV-27B-VL` 改为 `autotrust/JEV-27B-VL`，`Jev 1.13` 改为 TypeSafe Jev v1.13。Roadmap 去掉「训练 Qwen3.5-9B」，改为训练 Full-finetune 的 Qwen3.8-27B。

## v0.2.0 (2026-10-16)

- 发布 HF 模型 [shibing624/train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora) 和 [shibing624/train-jev-27b-lora](https://huggingface.co/shibing624/train-jev-27b-lora)（LoRA + 决策头 + 温度）。train-jev-27b-lora 在同题评测上准确率追平 JEV-27B-VL，ECE 0.053 对 0.097。
- `--adapter` 可以直接传 HF repo id，会一并下载 `head.pt` 和 `calibration.json`；`--adapter_revision` 固定版本。
- README 重写；部署与推理细节移到新的 `docs/serving.md`。`docs/` 下的文档默认中文，英文版为 `*.en.md`。
- 新增示例：`examples/local_inference.py`、`batch_decisions.py`、`wide_choices.py`、`smoke_systemone.py`。
- 训练脚本支持 `RESUME` / `SAVE_STATE_EVERY` / `SEED` / `STEPS`；`--resume` 找不到训练状态时直接报错，不会从头重训。
- `scripts/prepare_data.sh` 按发布模型所用的版本下载语料和底座；`scripts/di_sample.py` 的 Decision Index 抽样可复现。
- 删除 `patches/decision-index-resumable-download.patch`。

#### features

- `--full_ft` 全参训练（`trainjev/train_full.py`，FSDP2：fp32 主权重、梯度和 AdamW 状态按卡切分，bf16 计算；`--cpu_offload` 把分片放到主机内存）。各 rank 先在 meta 设备上建模型并切分，只有 rank 0 读 checkpoint、逐张量广播进分片，rotary `inv_freq` 重新计算；4B 实测加载阶段主机内存峰值 96 GiB → 31 GiB。全参 checkpoint 存文本模型权重（`model.safetensors`）+ `head.pt`，`load_trained` 统一加载 LoRA / 全参 checkpoint（`evaluate`、`eval_vision`、`serve` 共用）。4B 全参与 LoRA 的对照结果见 `docs/results.md` §9（全参未胜出，27B 用 LoRA）。
- 断点续训：`--save_state_every N` 每 N 步把可训练权重、AdamW 状态、各 rank RNG、step / best / history 用 `torch.distributed.checkpoint` 写到 `<out>/state`（全参按分片存，先写 `state.tmp` 再原子替换），`--resume` 接着训，数据顺序不变；换 GPU 数续训会直接报错。4B LoRA 实测"3 步 + 存档 + 新进程续 3 步"和直接跑 6 步的 adapter / head 完全一致。
- 训练内循环提速：loss 改为整个 micro-batch 一次算（`losses.kl_rps_batch`，24 槽位 logits + mask），不再每行一次 `.item()`；批量分词 `data.encode_many`（约 10×），下一步的采样和分词在后台线程预取；验证集只在启动时编码一次，`evaluate.py` 去掉重复编码。训练语义不变（采样、置换、归一化、学习率表、checkpoint 格式都没动）。GPU 上 3 步冒烟和改前不是逐位一致（bf16 / TF32 求和顺序），见 `docs/training.md`。
- `scripts/audit_training_data.py`：加新数据前审计 JEV 格式训练集——target / 槽位检查、归一化后重复与标签冲突、与验证 / OOD / PK / Decision Index 输入的 state 级和完整 prompt 级重叠，记录文件 SHA256；只报告，不自动删除或批准数据。
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
