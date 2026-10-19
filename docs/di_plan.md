# 追上 Decision Index：Instant / AutoThink 方案

[English](di_plan.en.md)

这份文档只使用已经公开的数字。本仓库这次改动**没有**在 Decision Index 上重跑出新分数，也没有把 StartLux 的公开子集和 train-jev 的 37 基准面板相减。两边协议不同，差值没有意义。

## 差距（引用，不是新测量）

train-jev 已发布结果（本仓库 README / CHANGELOG，公开 37 基准文本面板，110,087 题；采样为 6,000 题）：

| 模型 | DI 0.3 采样 | DI 0.3 全量文本面板 |
|---|---:|---:|
| train-jev-4b-lora | 41.5 | 39.01 |
| train-jev-27b-lora | 56.1 | 55.58 |
| TypeSafe Jev v1.13（官方 API，仅采样） | 59.7 | – |

StartLux-Decision-v1.1 自己报告的数字（[仓库](https://github.com/StartLuxLabs/StartLux-Decision)、[论文](https://arxiv.org/abs/2610.17309)、27B AutoThink [模型卡](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink)）。他们在公开子集上选模型，不是本仓库的独立复测：

| 模型 | DI 0.3 公开子集 | DecisionPivot 成对准确率（7,800 对，门控 0.9） |
|---|---:|---:|
| 27B AutoThink | 73.37 | 86.9 |
| 27B Instant | 72.83 | 75.4 |
| 9B AutoThink / Instant | 70.45 / 69.96 | 59.9 / 47.5 |
| 4B AutoThink / Instant | 68.21 / 67.88 | 59.9 / 41.5 |
| 2B / 0.8B Instant | 59.14 / 53.75 | – |

DI 0.3.2 榜（StartLux README，2026-10-16）：27B Instant Full 69.12 / 174（公开 72.88，私有同技能 64.82，私有新领域 61.41）。私有集不在本仓库。

他们论文里，双向注意力相对因果的受控增益最大约 **+2.4** DI 点，种子噪声约 0.2–0.7。AutoThink 相对 Instant 的 DI 增益大约 **0.3–0.5** 点；DecisionPivot 成对准确率才是 AutoThink 的主要收益（约 +11.5 到 +18.4 个百分点）。RL 主要缩短思维链（长度最多约 −65%，正常结束最多约 +48 个百分点）。

因此 DI 的大头差距来自数据覆盖和全参数训练，不是再加一个门控。

## 两边做法

| | train-jev（已发布） | StartLux Instant | StartLux AutoThink |
|---|---|---|---|
| 底座 | Qwen3.5-4B / Qwen3.8-27B | Qwen，0.8B–27B | Qwen，4B / 9B / 27B |
| 读出 | 24 槽位决策头，bare-v1 prompt，最后 token | `lm_head` 选项字母 logits，chat 模板 | 同 Instant，思维后再读一次 |
| 注意力 | 因果 | 仅 full-attention 层对 prompt 双向；线性层仍因果 | 全程因果，否则无法生成 |
| 训练 | LoRA r16 + 决策头；KL + 0.5 RPS；30% 选项置换 | 全参数软 CE，再在 Lux-Rounds 上训 adapter 并平均多种子 | 思维监督，再 RLCD² |
| 不确定时 | 一次前向 | 拒绝 `think` | top 概率低于门控才写思维；最终 `(1-mix)*p0 + mix*p1`，mix=0.75 |

本仓库**不**把 prompt 换成他们的 chat 模板，也**不**把读出换成字母 logits。已发布 checkpoint 仍按 bare-v1、因果、一次前向使用。缺 `thought` 时 prompt 与旧版逐字节相同。

双向只作用在 Qwen 的 full-attention 层（大约每 4 层一层）。未加 padding 时 `create_causal_mask` 返回 `None`，因果性由模块的 `is_causal` 决定，所以双向训练必须改这个开关，并且在反向重算时保持开启。把因果权重只在推理时改成双向会打乱预测，服务端拒绝这样做。

## 优先级

1. **数据 + 全参数（已有入口，未在本 PR 重训）**。`--full_ft` 走 FSDP2。StartLux Stage 1 用约 270 万（0.8B–9B）或约 80 万（27B）条样本，并全参数更新含输出投影。这是 DI 差距的主体。
2. **Instant：训练时双向（本 PR）**。他们的消融上限约 2.4 点，不是 30 点。必须用这个 mask 训练。
3. **多种子平均（本 PR，只平均 LoRA + head）**。他们 0.8B 平均 3 个种子，2B–27B 平均 4 个。全参数 safetensors 不在这个脚本里。
4. **AutoThink 服务 + 思维丢弃 + 本地对比对（本 PR）**。`thought_drop` 默认 0.5，让一次前向的读出继续被训练。本地 `precision-mix` 只覆盖退款、截止日期、计数、求和四族，**不是** DecisionPivot，也不是 DI 分数。开发集不要拿去训练或调 DI。
5. **RLCD² 目标函数（本 PR 只有库，没有训练循环）**。奖励是思维前后决策加权得分之差，再减去超出软预算（论文 B=3072，生成上限 4096）的长度，λ=1e-4；短于 8 token 奖励为 0。组内优势只中心化并截断到 [−5, 5]，不做标准差归一。在线 GRPO 需要生成、参考模型和 GPU，这里没有跑。
6. **下一轮，本 PR 没做**：none-of-the-above、他们的 chat 模板、分数题 2–10 档、超过 26 项的淘汰赛、DI 0.3.2 私有集。

## 怎么训、怎么评

默认训练仍是因果注意力。Instant / AutoThink 用单独脚本打开。

```bash
# 本地对比对。data/ 被 gitignore，不要提交生成的 jsonl。
python scripts/build_precision_mix.py --pairs 50 --out data/precision-mix

# Stage 2 Instant：从已有因果 checkpoint 续训，full-attention 双向。
INIT_ADAPTER=runs/jev-4b/best MIX=data/precision-mix/train.jsonl MIX_FRACTION=0.25 \
  LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 OUT=runs/jev-4b-instant \
  bash scripts/run_train_instant.sh

# 多种子平均（注意力模式、verbalizer、LoRA 配置必须一致）
python scripts/average_lora.py --out runs/jev-4b-instant-avg \
  runs/seed0/best runs/seed1/best runs/seed2/best runs/seed3/best

# Stage 3 AutoThink：保持因果，一半思维行被丢掉。
INIT_ADAPTER=runs/jev-4b/best MIX=data/precision-mix/train.jsonl MIX_FRACTION=0.25 \
  LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 OUT=runs/jev-4b-autothink \
  bash scripts/run_train_autothink.sh
```

全参数把 `--full_ft` 加到 `trainjev.train`（或继续用 `trainjev.train_full`）。27B 把脚本换成 `run_train_27b.sh`，并自己设 `ATTENTION=bidirectional` 或 `THOUGHT_DROP`。

服务：

```bash
# 已发布模型：不传 VARIANT，行为和以前一样
bash scripts/run_serve.sh

# 只有 head.pt 里 attention=bidirectional 的 checkpoint 才能用 instant
VARIANT=instant CKPT=runs/jev-4b-instant/best bash scripts/run_serve.sh

# 因果 checkpoint。默认门控 0.6；复现他们公布的数字时用门控 0.9
VARIANT=autothink CKPT=runs/jev-4b-autothink/best bash scripts/run_serve.sh --gate 0.9
```

`POST /v1/decisions` 与 `/v1/systemone` 是同一个处理函数。`think_mode` 取 `auto` / `never` / `always`（别名 `hybrid` / `plain` / `think`）。plain 和 instant 只接受 `never`。带图的题和超过 16 个选项的选择题不写思维。思维 prompt 会超过 `--max_len` 时保留第一次读出。

评测仍走现有 `evaluate.py` / `run_eval.sh`，读出使用 checkpoint 里的 `attention`。不要在 `precision-mix/dev.jsonl` 或最终 DI 题上拟合温度。在冻结的同一套 DI 上重跑之前，不要写新的指数。成对准确率用 `trainjev.precision.pair_accuracy` 只报告本地对比对。

## 风险

- 因果 checkpoint 在服务时被改成双向：已拒绝。反向把双向权重量成因果同样没有意义。
- LoRA 没在「带思维的 prompt」上训过时，采样出的思维可能没用，第二次读出不应被当成收益。`thought_drop` 保留无思维样本。
- 思维采样不确定。`return_thought` 只用于检查，不进入分数。
- 视觉请求和宽选择题跳过思维。
- 多种子平均不覆盖全参数权重。
- RLCD² 库有单测，没有在线训练，也没有新的 DI 或 DecisionPivot 数字。
