# 更新日志

## [Unreleased]

#### serving

- `/v1/systemone` 新增 TypeSafe 风格图片 assets：JSON/base64 与 multipart，引用支持 state / 单题 instructions / criteria，保留对象键与题目隔离；本地请求不需要额外实验 header。
- 新增 `--vision` 完整 VL 加载路径，与纯文本共享已合并 LoRA 的语言模型及 FP32 head；三种题型和原有宽选项策略均可读图，超长图片 prompt 拒绝而不截断。纯文本默认加载/读法保持不变；proxy 转发图片保留所有位置的引用。
- 补齐图片/JSON/上传体积及格式校验、CPU 路由与像素影响测试、中英接口示例和 multipart 依赖；暂不支持 Box、URL assets 或压缩请求，不以接口支持声称视觉准确率提升。
- 增加 `VISION=1` 启动方式和 `examples/image_decisions.py`；以小型随机 Qwen VL 模型验证真实视觉编码器、图像处理器、LoRA 加载及校准读出。CI 使用 CPU torchvision，`cuda:0` 保持 bf16 推理精度。
- 取消固定的 280–1568 边长限制，采用 Qwen 图像处理器缩放；支持常见光栅格式，动图/多页图取第 0 帧。保留请求体积、解码像素与上下文资源保护。

#### documentation and data

- 重写中英文训练文档：从 Qwen 底座开始，介绍公开训练集、蒸馏、金标续训、校准、耗时和结果，删除未公开构建流程的细节。
- `run_serve.sh` 默认加载 HF 模型；训练脚本支持 `INIT_ADAPTER`、`MIX`、`MIX_FRACTION`、`LR` 和 `HEAD_LR`；`run_eval.sh` 默认不重新校准已发布模型。
- 发布 `data/train-jev-pk`（2,494 题）和 `data/train-jev-classification-pk`（1,300 题）的完整索引、标签与可分发输入，新增恢复和本地评测脚本。未核验许可的原文不包含在仓库中；缺失输入不会被计作完整评测。

#### results

- 完成 v0.2.0 两模型的 Decision Index 0.3 公开 37 基准可评分文本全量：各 110,087 请求 / 272,229 子问题，全部成功，coverage=1.0；4B 指数 37.40、27B 53.88。包含 HLE 501 个文本请求和 HoVer 4,000 个请求，不是多模态 HLE 或含私有测试的 Full 榜单。
- 新增中英全量报告、协议/分项/指纹小 JSON；保留历史 35 采样及新 37 quick6000（38.14/53.29），不把不同输入/读法差值解释为训练增益。全量使用冻结源码、HF merged bf16/fp32 head、single 和各模型原温度，没有新训练或温度重拟合；4B 暂停后复用 2,882 条成功结果续跑，最终两模型都已完成并释放评测模型。

#### fixes

- README / README_EN 对照表里的官方模型名称不准确：`JEV-27B-VL` 改为 `autotrust/JEV-27B-VL`，`Jev 1.13` 改为 TypeSafe Jev v1.13。Roadmap 去掉「训练 Qwen3.5-9B」，改为训练 Full-finetune 的 Qwen3.8-27B。

## v0.4.0 (2026-10-23)

- 更新 4B / 27B LoRA 默认权重，公开 DI37 全量得分为 42.00 / 57.01，train-jev-pk 宏平均准确率为 76.03% / 79.52%。保留旧权重版本，不重跑旧模型或重新拟合温度。
- 披露退步：27B transfer-v4 准确率 85.82% → 84.76%、ECE 0.0316 → 0.0681；4B 分类与 transfer 的 ECE 也变差。旧分类宽选项使用 tournament，新权重使用 single，不把所有差异归因于训练。
- 双向注意力实验已删除；本次权重均为因果模型。更新中英文 README、当前与历史结果标签以及完整性/来源限制，不新增实验功能。

## v0.3.0 (2026-10-18)

- **27B v0.3.0**：同一配方（同数据、12.5% 新监督、学习率 2e-5 / 4e-5、seed 17、2,048 步学习率表）应用到 Qwen3.8-27B，以 v0.2.0 27B 为起点续训 1,024 步（3×H20，7.9 h），开发集选第 512 步。封存 2,494 题同题 PK 宏平均 73.99% → 78.72%（配对 +4.73pp [3.71, 5.80]），**对 autotrust/JEV-27B-VL +5.25pp [4.01, 6.50]**，公开金标 / 旧分类 / 规则三类都领先；transfer-v4 0.851 → 0.858、ECE 0.053 → 0.032；13 数据集 PK 0.678 → 0.687（p=0.073）；Decision Index 历史 35 基准采样 54.57 → 56.15（autotrust/JEV-27B-VL 54.11），37 基准全量 53.88 → 55.58（配对题级 +1.03pp [0.86, 1.20]，30 升 / 6 降）。
- 4B v0.3.0 Decision Index 37 基准全量 37.40 → 39.01（配对 +0.97pp [0.78, 1.17]，3 个基准显著下降）；历史 35 采样 40.35 → 41.49；transfer-v4 0.779 → 0.805。
- README / README_EN 的模型与效果表改为 v0.3.0 4B + 27B；`docs/results*.md` 新增 §13；`reports/v0.3.0/external_final2.json` 扩为六模型（含 27B v0.2.0 / v0.3.0）。已发布到 [shibing624/train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora) 和 [shibing624/train-jev-27b-lora](https://huggingface.co/shibing624/train-jev-27b-lora)：main 与 `v0.3.0` tag 为新权重，旧权重保留在 `v0.2.0` tag。
- 补齐 27B DI37 全量与发布模型卡：三卡互斥分片合并后 110,087 请求 / 272,229 答案字段全部成功，40 条跨 GPU 答案含概率精确一致；换旧温度 0 翻转。端到端 7.93 h，不作为单 GPU 加速证据。新增 `reports/v0.3.0/di37_results.json`；诊断区间不作为 DI 指数显著性，保留 Chess/HLE 退步及 ACOS 指标差异，无 autotrust/JEV-27B-VL 全量配对。
- 新训练组合：旧 Jev 分布蒸馏回放 + GSM8K 双干扰项视图、PAWS、AQuA-RAT、QASC、HelpSteer2 和双算法校验的规则数据。以 v0.2.0 LoRA/决策头续训，每步 12.5% 新监督；三卡完整训练 2,048 步，开发集保护条件选择 512 步 checkpoint。
- 同题最终 PK 使用第二批封存的 2,494 条未训练样本，22 个任务宏平均准确率 63.59% → 74.18%，增益 +10.60pp（95% group-bootstrap 区间 [9.44, 11.77]）。公开金标宏平均 57.03% → 63.25%；旧任务 70.22% → 69.89%，宽选项持平；保留退化及局限，不将采样 PK 当作 Decision Index 分数。
- 校准后宏平均 NLL 1.050 → 0.751、Brier 0.496 → 0.350；同 H20 充分预热、模型顺序复验后的 p50 41.42 / 40.88ms，开销基本持平。第一候选宽选项保护失败后未发布，首次最终集未用于修复训练。
- 同一批 2,494 条上按各自原生推理协议补测 Kev-4B 与 autotrust/JEV-27B-VL（不重训、不重拟合温度）：任务宏平均 v0.3.0 74.18% / Kev-4B 63.50% / autotrust/JEV-27B-VL 73.47%。对 Kev-4B 配对 +10.69pp [8.89, 12.59]；对 autotrust/JEV-27B-VL +0.71pp [−0.82, 2.34]，统计持平，规则任务领先、旧分类任务落后 2.78pp。明细 `reports/v0.3.0/external_final2.json`。
- 当前训练入口新增 `--init_adapter`（保留可训练 LoRA/head，优化器重置）、`--mix_fraction`（同类型定额替换）、`--kind_counts` 与实际采样/token 统计。新增可选 `--brier_w`，默认 0，本版最终配方未开启；旧默认损失和推理接口保留。
- 本地发布包包含可加载的 adapter/head/calibration、无内部路径的训练来源/真实指标和完整性清单；原 v0.2.0 权重保留在 HF 的 `v0.2.0` tag，不覆盖。memory-write 专项与派生记忆生成单列为 v0.4.0 待办。

## v0.2.0 (2026-10-16)

- 发布 HF 模型 [shibing624/train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora) 和 [shibing624/train-jev-27b-lora](https://huggingface.co/shibing624/train-jev-27b-lora)（LoRA + 决策头 + 温度）。train-jev-27b-lora 在同题评测上准确率追平 autotrust/JEV-27B-VL，ECE 0.053 对 0.097。
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
- 超过 16 个选项的 `choice` 题新增 `single` 和 `permute` 两种读法，和原有的 `tournament` 一起，对应 autotrust/JEV-27B-VL `serve_decide.py` 的三种策略。服务端用 `--strategy` 选（默认 `single`，和 autotrust/JEV-27B-VL 在 Qwen 底座上的默认一致），单个请求也可以在请求体里传 `"strategy"` 覆盖。`single` 一次前向读完：A–P 用训练过的决策头，Q–Z、AA、AB… 用底座 `lm_head` 的行、bias 为 0（`trainjev.modeling.choice_labels`）。`tournament` 现在把各组放在一个 batch 里打分。
- `trainjev.serve --device cpu`：没有 GPU 时用 fp32 在 CPU 上推理（很慢，只适合小规模核对）。
- 宽选项意图分类评测：`scripts/wide_build.py` 从 HF Hub 构建 CLINC150（150 类）/ BANKING77（77 类）/ MASSIVE en（59 类），后两个和 PK 题库逐字节一致；`scripts/run_wide.sh` 对三种策略各跑一遍并输出配对报告。`scripts/pk_benchmark.py run` 新增 `--strategy` 和 `--log_every`。

#### fixes

- README 里"`choice` 2–16 个选项，>16 个用锦标赛扩展到 256"的说法不准确：autotrust/JEV-27B-VL 在 Qwen 底座上默认是一次前向读 256 个标签，锦标赛是它在 Gemma 底座上的默认做法。README / README_EN / docs 已改写。
- 仓库不再作为 Python 库打包：去掉 `pyproject.toml` 里的 setuptools 配置，不再支持 `pip install -e .` 以及 `.[serve]`、`.[dev]` 这些 extras。依赖只通过 `requirements.txt` 安装，在仓库根目录用 `python -m trainjev.train` 训练。
- README 方法图的虚线边写成了 `-.KL + 0.5·RPS.->`，`+` 不在引号里，GitHub 的 Mermaid 直接报 Lexical error，整张图渲染不出来。边标签改成 `-.->|"KL + 0.5 RPS"|`。
- 删掉 README / README_EN 里的 Project Structure 目录树。
- 推理服务只保留 TypeSafe System One API：去掉服务端的 `/v1/decide` 路由（JEV `serve_decide.py` 的原生接口，仓库里没有任何代码调用）。对外接口只有 `POST /v1/systemone` 和 `GET /health`。`--proxy` 模式和 `trainjev.evaluate` / `trainjev.eval_vision` 的 `--decide_url` 不变，它们作为客户端去调 autotrust/JEV-27B-VL 自己的服务，用来做对比。

## v0.1.0 (2026-10-14)

首次发布。

- 在 Qwen3.5 上复现 JEV System 1 配方：全部 Gated DeltaNet / attention / MLP 投影加 LoRA r16，24 槽位 fp32 决策头从 `lm_head` 初始化，损失为 KL + 0.5·RPS，30% 选项置换，按类型分层的 128 行 batch，多卡 DDP（`trainjev.train`）。
- 按问题类型做温度校准，并在语料上评测（`trainjev.evaluate`）。
- 兼容 TypeSafe System One 的服务（`/v1/systemone`），支持超过 16 个选项的锦标赛，以及 autotrust/JEV-27B-VL 的代理模式（`trainjev.serve`）。
- Decision Index 视觉集上的零样本图像决策评测（`trainjev.eval_vision`）。
- Decision Index 0.3 辅助脚本：分层抽样和只含样本的 suite（`scripts/di_sample.py`），丢掉部分基准后重算榜单（`scripts/di_index.py`）。
- 13 个数据集的分类 PK：从 HF Hub 构建逐字节一致的 suite（`scripts/pk_build.py`），以及可断点续跑、带配对报告的评测（`scripts/pk_benchmark.py`）。
- Qwen3.8-27B 底座已端到端跑通；脚本 `scripts/run_train_27b.sh`。
- Python >= 3.12。
- Qwen3.5-4B 的结果见 [docs/results.md](docs/results.md)。
