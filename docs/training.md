# 从 Qwen 底座训练 train-jev

[English](training.en.md)

本仓库用 **LoRA + 24 槽位决策头**训练能返回概率的决策模型。完整路线是：从公开 Qwen 底座开始，用 Jev 目标分布蒸馏，再加入金标与规则数据增强判断能力，最后校准并部署。

你可以直接使用 [train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora) / [train-jev-27b-lora](https://huggingface.co/shibing624/train-jev-27b-lora)，也可以按下面的流程训练自己的模型。

## 1. 底座与硬件

| 规模 | 底座 | LoRA + 决策头 | 参考训练硬件 |
|---|---|---:|---|
| 4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 30.5M + 61k | 3×H20 |
| 27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 108.8M + 123k | 4×H20 |

底座权重冻结，只更新 LoRA 和决策头；视觉编码器不参与训练。4B 启动脚本默认关闭 gradient checkpointing，27B 默认开启。显存不足时减小 `MICRO_TOKENS`，或用单独的训练命令开启 gradient checkpointing。

```bash
git clone https://github.com/shibing624/train-jev.git && cd train-jev
pip install -r requirements.txt
```

需要 Python 3.12+、支持 Qwen3.5 的 transformers 和 CUDA。`requirements.txt` 包含线性注意力加速所需的 `flash-linear-attention`。

## 2. 训练集

| 数据 | 用途 |
|---|---|
| [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | 655,806 行软分布和程序化 target，用于基础蒸馏和回放 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat)、[QASC](https://huggingface.co/datasets/allenai/qasc) | 数学与科学选择题 |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) | 释义是否一致的二分类 |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 将回答的 correctness 0–4 转成五选一 |
| 业务规则或自己的标注数据 | 路由、证据判断、日期计算、退款等任务 |

下载基础蒸馏数据和底座：

```bash
bash scripts/prepare_data.sh
# 训练 27B 时下载对应底座
BASE_REPO=Qwen/Qwen3.8-27B BASE_DIR=models/Qwen3.8-27B bash scripts/prepare_data.sh
```

新增数据统一转成 JSONL，每行是一道题，`target` 与 `options` 一一对应：

```json
{"kind":"choice","state":"客户的卡被重复扣款。","question":"哪个团队处理？","options":["售后","物流","财务"],"target":[0,0,1],"weight":1.0}
```

- `noul`：`options` 固定为 `["false","true"]`，`target` 为两个概率。
- `choice`：2–16 个选项，可以用 one-hot 金标，也可以用教师的软分布。
- `score`：6 个有序等级 `["0",...,"5"]`；其他等级数转换为 `choice`。

下面是把公开 PAWS 转成训练行的最小示例，保存为 `data/my-gold/train.jsonl`：

```python
import json
from pathlib import Path
from datasets import load_dataset

dataset = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
output = Path("data/my-gold/train.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as stream:
    for row in dataset:
        if row["label"] not in (0, 1):
            continue
        item = {
            "kind": "noul",
            "state": f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
            "question": "Do the two sentences have the same meaning?",
            "options": ["false", "true"],
            "target": [1 - row["label"], row["label"]],
            "weight": 1.0,
        }
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
```

PAWS 示例可用于复用训练方法，但不等于发布模型使用的全部金标与规则组合，因此不要直接引用发布模型的评测分数。

## 3. 基础蒸馏：从底座开始

```bash
GPUS=0,1,2 bash scripts/run_train.sh
# 或训练 27B
GPUS=0,1,2,3 bash scripts/run_train_27b.sh
```

默认 global batch 128、LoRA r16 / alpha32、LoRA 学习率 1e-4、决策头学习率 2e-4。约 5,123 步完成基础蒸馏；输出位于 `runs/jev-4b/` 或 `runs/jev-27b/`，其中 `best/` 是验证 KL 最低的 checkpoint。

训练方法：

- 决策头从底座 `lm_head` 的 `false/true`、`0–5`、`A–P` 对应行初始化，不从随机分类器起步。
- `KL(target || model)` 学习完整目标分布；`score` 题加 0.5×RPS，使远离目标等级的错误更受惩罚。
- 30% 的选择题随机打乱选项，target 同步置换，减少位置偏好。
- AdamW，3% warm-up 后 cosine 衰减，梯度裁剪为 1.0。

## 4. 金标与规则续训

以刚训练的 checkpoint 为起点，混入自己的 JSONL 数据，继续训练 LoRA 和决策头：

```bash
GPUS=0,1,2 INIT_ADAPTER=runs/jev-4b/best MIX=data/my-gold/train.jsonl \
  MIX_FRACTION=0.125 LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 SEED=17 \
  KIND_COUNTS=choice:64,noul:32,score:32 OUT=runs/jev-4b-supervised \
  EVAL_EVERY=256 bash scripts/run_train.sh
```

27B 用对应的起点和脚本：

```bash
GPUS=0,1,2 INIT_ADAPTER=runs/jev-27b/best MIX=data/my-gold/train.jsonl \
  MIX_FRACTION=0.125 LR=2e-5 HEAD_LR=4e-5 STEPS=2048 MAX_STEPS=512 SEED=17 \
  KIND_COUNTS=choice:64,noul:32,score:32 OUT=runs/jev-27b-supervised \
  EVAL_EVERY=256 bash scripts/run_train_27b.sh
```

多个文件用逗号连接，例如 `MIX=data/math/train.jsonl,data/public/train.jsonl,data/rules/train.jsonl`。旧数据回放降低遗忘，新监督增强任务能力；发布配方采用 12.5% 新监督。`STEPS` 控制学习率表，`MAX_STEPS` 控制停止位置，两者用途不同。

发布的 v0.3.0 模型使用上述两阶段方法和 37,147 行新增监督，并选用第 512 步的续训权重。用公开来源自行处理数据可以复用配方，但当前仓库未分发发布时的完整加工数据与生成器，不保证逐位复现原权重。

## 5. 校准、续训与部署

自己训练的 checkpoint 可以在独立 `calibration.jsonl` 上拟合各题型温度：

```bash
CKPT=runs/jev-4b-supervised/step00512 FIT_CALIBRATION=1 bash scripts/run_eval.sh
# 27B
BASE=Qwen/Qwen3.8-27B CKPT=runs/jev-27b-supervised/step00512 \
  FIT_CALIBRATION=1 bash scripts/run_eval.sh
```

上面的命令默认使用蒸馏数据的 calibration；新业务域应准备自己的独立校准数据，并通过 `DATA` 指定。不要在最终评测题上拟合温度。已发布 HF 模型自带校准，`bash scripts/run_eval.sh` 默认只评测，不重新拟合。

长时间训练中断后，使用相同配置和输出目录，加 `RESUME=1`：

```bash
GPUS=0,1,2 RESUME=1 OUT=runs/jev-4b bash scripts/run_train.sh
```

`SAVE_STATE_EVERY` 控制存档间隔。`best/` 用于推理，`state/` 用于恢复优化器和随机数状态；恢复时保持 GPU 数和训练配置不变。

启动自己训练的模型：

```bash
CKPT=runs/jev-4b-supervised/step00512 bash scripts/run_serve.sh
# 直接体验发布模型
bash scripts/run_serve.sh
```

更多推理参数见 [部署文档](serving.md)。

## 6. 实测耗时与训练结果

| 训练 | 硬件 | 步数 | 耗时 |
|---|---|---:|---:|
| 4B 基础蒸馏 | 3×H20 | 5,123 | 2.77 h |
| 27B 基础蒸馏 | 4×H20 | 5,123 | 33.07 h |
| 4B 金标与规则续训实验 | 3×H20 | 2,048 | 1.3 h |
| 27B 金标与规则续训实验 | 3×H20 | 1,024 | 7.9 h |

续训实验耗时是实际跑完的实验总时间；发布模型使用其中第 512 步，并非最后一步。

| 已发布模型 | [train-jev-pk](../data/train-jev-pk) 准确率 ↑ | [transfer-v4](https://github.com/jaredpalmer/kev) 准确率 ↑ | transfer-v4 ECE ↓ | [Decision Index 全量](https://github.com/apolinario/decision-index) 得分 ↑ |
|---|---:|---:|---:|---:|
| shibing624/train-jev-4b-lora | 74.18% | 0.805 | 0.054 | 39.01 |
| shibing624/train-jev-27b-lora | 78.72% | 0.858 | 0.032 | 55.58 |

这些是发布模型在固定评测协议下的结果，不是 PAWS 示例训练的预期分数。与 [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)、Kev 和 TypeSafe API 的对比及局限见 [结果文档](results.md)。

## 7. Instant 与 AutoThink

默认仍是因果注意力和一次前向，已发布权重不用改。要对齐 StartLux-Decision 的 Instant，用 `scripts/run_train_instant.sh`（full-attention 层双向，线性层仍因果）。AutoThink 用 `scripts/run_train_autothink.sh`（保持因果，`THOUGHT_DROP` 默认 0.5）。多种子平均见 `scripts/average_lora.py`。本地对比对 `scripts/build_precision_mix.py` 不是 DecisionPivot，也不是 DI 分数。步骤、已引用的公开数字和风险见 [Decision Index 方案](di_plan.md)。本仓库还没有在同一套 DI 上测出新分数。
