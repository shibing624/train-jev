# 训练

[English](training.en.md)

本文说明 `trainjev/train.py` 实现的 JEV System 1 训练配方，以及每一步为什么这样做。

## 0. 快速开始

```bash
bash scripts/prepare_data.sh                                    # data/jev-distill-corpus-v3 + models/Qwen3.5-4B
BASE_REPO=Qwen/Qwen3.8-27B BASE_DIR=models/Qwen3.8-27B bash scripts/prepare_data.sh   # 下载 27B 底座

GPUS=0,1,2 bash scripts/run_train.sh                            # 4B，3 张 H20，约 2.8 小时
GPUS=0,1,2,3 bash scripts/run_train_27b.sh                      # 27B，4 张 H20，约 33 小时
GPUS=0 MAX_STEPS=3 OUT=runs/smoke bash scripts/run_train.sh     # 冒烟测试（学习率表仍按 5,123 步）
GPUS=0,1,2,3 RESUME=1 OUT=runs/jev-27b bash scripts/run_train_27b.sh   # 断点续训

CKPT=runs/jev-4b/best bash scripts/run_eval.sh                  # 拟合温度并评测
BASE=models/Qwen3.8-27B CKPT=runs/jev-27b/best OUT=eval/jev-27b bash scripts/run_eval.sh
```

启动脚本读取这些环境变量：`GPUS`、`BASE`、`DATA`、`OUT`、`SEED`、`STEPS`、`MAX_STEPS`、`MICRO_TOKENS`、
`EVAL_EVERY`、`SAVE_STATE_EVERY`（默认 100）和 `RESUME`。`prepare_data.sh` 按发布模型训练时用的版本下载语料和底座，
访问不了 Hugging Face 时设置 `HF_ENDPOINT` 走镜像。

| 实测训练 | 单卡显存峰值 | 耗时（含周期验证） |
|---|---:|---:|
| 4B，3×H20，关闭 gradient checkpointing，`--micro_tokens 4096` | 48.8 GB | 2.77 h |
| 27B，4×H20，开启 gradient checkpointing，`--micro_tokens 16384` | 74.9 GB | 33.07 h（23.2 s/step） |

## 1. 数据格式

每行训练数据是关于一段 state 的一道带类型的题：

```json
{"kind": "choice", "state": "...", "question": "Which team should handle this?",
 "options": ["billing", "infra: timeouts and capacity", "product"], "target": [0.05, 0.90, 0.05],
 "source": "yuri_v3", "weight": 1.0}
```

- `noul`：选项为 `["false", "true"]`，target 为 `[P(false), P(true)]`
- `score`：选项为 `["0", ..., "5"]`，target 是 6 个有序等级上的分布
- `choice`：2–16 个选项，target 与 `options` 一一对应

`target` 是完整的概率向量：语料里既有教师模型的软分布，也有程序化生成的标签，并不都是 one-hot，也不都来自同一个 API。
其他 jsonl 文件按同样格式即可用 `--mix path.jsonl:N` 混入（重复 N 次），`weight` 缩放该行的损失。
加入新数据前先审计 target 和来源（见下文"数据审计"）。

## 2. Prompt 模板（`bare-v1`）

```
[kind] choice
[state] Ticket #4821 ...
[question] Which team should handle this?
[options]
A) billing
B) infra: timeouts and capacity
C) product
[decision]:
```

`noul` 和 `score` 直接列出选项原文（`false`、`true`；`0` .. `5`）。模板与 JEV-27B-VL 的 `serve_decide.py` 逐字节一致，
所以这里训出的模型可以和 JEV 在完全相同的输入上对比。不加 chat template，也不加特殊 token。state 过长时从中间截断
（保留头尾，中间用 ` … ` 连接），question 和选项永远不截断。

## 3. 模型

```
Qwen3.5 文本模型（冻结，bf16）+ LoRA  ──► 最后一个 prompt token 的 hidden state ──► 24 槽位 fp32 决策头
```

**决策头。** 24 行，每行对应一个 verbalizer token：`false, true | 0..5 | A..P`。类型为 `k`、有 `n` 个选项的题使用从
`RANGES[k]` 开始的 `n` 个槽位。决策头用底座 `lm_head` 中这些 token 的行初始化，因此在第 0 步：

```
softmax(head(h)[slots]) == softmax(lm_head(h)[verbalizer_ids])
```

也就是说，未训练的模型恰好等于底座在合法答案上的零样本 next-token 判断。`scripts/check_init.py` 验证了这一点
（Qwen3.5-4B，fp32，最大 |Δp| 7.7e-6）。所以训练是从预训练已有的能力出发（`test_set_30k` 上 noul AUROC 0.82、
choice top-1 0.51），而不是从一个随机分类器开始。

**LoRA。** r=16、α=32、dropout 0.05，覆盖 Qwen3.5 混合架构里两种层的所有投影：

| 层类型 | 模块 |
|---|---|
| Gated DeltaNet（线性注意力） | `in_proj_qkv`、`in_proj_z`、`out_proj` |
| 全注意力 | `q_proj`、`k_proj`、`v_proj`、`o_proj` |
| MLP | `gate_proj`、`up_proj`、`down_proj` |

4B 有 30.5M 可训练 LoRA 参数和 61,464 个决策头参数；27B 分别是 108.8M 和 122,904。训练只用这个小决策头，
不用整个词表投影。底座文件本身不变，但启用或合并 adapter 后，由于编码器是共享的，生成行为可能改变；这一点没有评测。

**右 padding，不需要 mask。** batch 统一右 padding，在每行最后一个真实 token 处读 hidden state。因果模型里，
最后一个真实 token 之后的 padding 影响不到它；递推的 Gated DeltaNet 层也一样，位置 t 的状态只依赖 ≤ t 的 token。
如果用左 padding，pad token 会被送进递推状态。

## 4. 损失

```
loss = Σ_rows weight · [ KL(target ‖ softmax(z)) + 0.5 · RPS(target, softmax(z)) · 1{kind = score} ] / batch
```

- **对给定分布做 KL**：不仅要求 argmax 一致，也要求不确定性一致。实测拟合出的温度接近 1.0，但这既不是损失函数保证的，
  也不保证在每个新领域都成立。
- **RPS**（ranked probability score，累积分布之间的平方距离）只加在 `score` 题上：离目标等级越远的概率质量惩罚越重，
  而单用 KL 不区分"差一级"和"差五级"。

K=6 时 RPS 为 `sum((cumsum(p)-cumsum(t))[:K-1]**2)/(K-1)`，不含最后一个累积桶。损失的分母是全局 batch 的行数，
不是各行权重之和。target 先归一化再做向量化计算。决策头在 fp32 下计算，底座在 bf16 下计算。

## 5. 数据处理

- **占位行。** 训练集中 yuri_v1 来源有 137,203 行，大多是均匀的二元 target。现有启发式规则还会标记任何均匀的二选一
  target（训练集共标记 138,889 行）；这只是规则匹配，不代表标签错误率。这些行的损失权重降为 ×0.05（`--placeholder_w`），
  并且不参与温度拟合；`--placeholder_w 0` 会直接丢掉它们。
- **选项置换。** 30% 的 `choice` 题会打乱选项，target 同步置换（`--perm`）。不这样做，模型会学到位置偏好
  （JEV-27B 的测量显示，未训练的底座在选项重排后有 41% 的题会改答案）。
- **按类型分层采样。** 每个 128 行的 step 先每种类型各抽 floor(128/6)=21 行，再从全局池里抽满剩下的 65 行。
  每次抽样内部无放回，但分层抽样和全局抽样之间可能重叠，且各 step 独立重新抽样，所以这不是按 epoch 遍历。
  采样的随机数种子由 step 决定，因此所有 rank 看到同一个全局 batch，各取 `rank::world` 那一份。
- **超过 16 个选项** 的题不参与训练。推理时由 `trainjev/serve.py --strategy` 决定怎么读（对应 JEV-27B-VL
  `serve_decide.py` 的三种策略）：`single`（默认；一次前向，A–P 用训练过的决策头，Q–Z、AA、AB… 用底座 `lm_head`
  的行、bias 为 0）、`tournament`（每组 ≤16 个先分组打分，再进行决赛，只读训练过的 A–P）、`permute`
  （`single` 跑 4 种选项顺序后取平均）。见 [results.md §8](results.md#8-超过-16-个选项的选择题)。

### 加新数据前的审计

加入候选金标数据前先运行 `scripts/audit_training_data.py`：

```bash
python scripts/audit_training_data.py \
  --train data/jev-distill-corpus-v3/train.jsonl --gold data/candidate-gold.jsonl \
  --heldout data/jev-distill-corpus-v3/validation.jsonl data/jev-distill-corpus-v3/ood.jsonl \
  --pk-suite data/pk --di-rows data/di-requests.jsonl.gz --out outputs/data-audit
```

把所有最终评测的输入都传进去，也可以用 `--source-jsonl` 传入带上游来源信息的原始记录。脚本会检查 target 和槽位、
归一化后的重复和 target 冲突、state 级重叠和完整 prompt 级重叠，并记录文件 SHA256。它解析顶层 `state`、PK 的 `text`，
以及 `payload.body.state` / `body.state` / `request.state`；缺少的输入会报告为证据不完整。只含预测结果的
`rows.json` 不能代替 benchmark 的请求文件。

候选索引按归一化后的 state 分组，而不是按单道题分组，并且所有行一律标记为"未批准"：脚本不验证标签是否正确，
也不验证上游 split / 许可、来源 / 模板是否独立，或反事实成对样本是否被隔离开。只是大小写 / 空白不同的匹配需要人工复核，
不会自动删除。

在原始语料上的审计发现：训练集和分布内的切分大量共享 state，而 OOD 的 state 不重合。共享 state 不等于题目重复或标签泄漏，
同一段上下文本来就可以出多道题。两个层级都要报告；选择新配方时用 state / 分组都不重合的新开发集。
当前复现保留原有切分，以便对比。

全量文件审计结果（2026-10-15，不是评测用的小样本）：

| 切分 | 行数 | 归一化 state 出现在训练集中的行 | 完整 prompt 也在训练集中的行 |
|---|---:|---:|---:|
| validation | 14,111 | 12,225 | 0 |
| calibration | 13,766 | 11,873 | 0 |
| test | 14,261 | 12,373 | 0 |
| test_set_30k | 29,955 | 28,065 | 0 |
| ood | 13,058 | 0 | 0 |

训练池共 655,806 行，只有 1 个归一化后重复的 prompt，且该重复没有 target 冲突。约 21.2% 的行命中占位行启发式规则，
这不是对标签语义错误率的估计。第一批候选金标数据有 39,973 行结构合法的数据，与蒸馏训练池以及提供的留出集输入在
归一化 state 和完整 prompt 上都没有重叠；但这不覆盖尚未完成的完整 Decision Index 面板和缺失的 transfer 请求输入，
也不能证明来源 / 模板独立。候选索引按 state 分组为 36,218 行训练、3,755 行开发，全部仍未批准用于训练。

## 6. 优化

| 项 | 取值 |
|---|---|
| 优化器 | AdamW（fused），β = (0.9, 0.98)，weight decay 0 |
| 学习率 | LoRA 1e-4，决策头 2e-4 |
| 学习率表 | 前 3% 步线性 warm-up，之后一次 cosine 降到 0.02× |
| 梯度裁剪 | 全局 grad norm 1.0 |
| 步数 | 一个 epoch 等价：655,806 / 128 = 5,123 |
| micro-batch | 按长度排序，每卡 `行数 × 最大长度 ≤ --micro_tokens` 个 padded token |
| DDP | 所有 micro-batch 跑完后手动展平并 all-reduce 梯度，所以各 rank 的 micro-batch 数可以不同 |

每个 micro-batch 的损失在 `backward()` 之前除以 `batch / world`，因此 all-reduce 取平均后，无论行怎么在卡间分配，
梯度都等于全局 128 行 batch 上的平均。

## 7. Checkpoint 与温度校准

每 `--eval_every` 步在 `validation.jsonl` 的固定 `--val_n` 子集上评测一次，保存到 `stepNNNNN/`；验证 KL 最低的同时存到
`best/`。每个 checkpoint 包含：

```
adapter_config.json  adapter_model.safetensors   # 文本模型的 PEFT LoRA
head.pt                                          # {"weight": [24, d], "bias": [24], "verbalizer_ids", "base", "args"}
calibration.json                                 # 由 `trainjev.evaluate --fit_calibration` 写入
```

`trainjev.evaluate --fit_calibration` 在 `calibration` 切分上（排除占位行）为每种题型拟合一个温度：对 log T 做黄金分割搜索，
最小化与软 target 的交叉熵。推理服务会把 logits 除以这些温度。

### 断点续训

`best/` 和 `stepNNNNN/` 是**推理用的 checkpoint，不是完整的训练状态**。27B 这类长时间训练要从一开始就打开
`--save_state_every 100`：所有 rank 一起用 `torch.distributed.checkpoint` 把可训练权重、AdamW 状态、
Python / Torch / CUDA 随机数状态、已完成的步数、最佳验证记录和历史写到 `<out>/state/`。全参训练时模型和优化器状态按分片保存。

用同样的命令、输出目录和 GPU 数，加上 `--resume` 重启：

```bash
torchrun --nproc_per_node 4 -m trainjev.train \
  --base Qwen/Qwen3.8-27B --data data/jev-distill-corpus-v3 --out runs/jev-27b \
  --micro_tokens 16384 --save_state_every 100 --resume
```

模型、数据、种子、全局 batch、学习率、数据增强和总步数 `--steps` 都要保持不变。`--max_steps` 只是提前停止的上限，
不会缩短学习率表。GPU 数不同时会直接报错，因为每个 rank 分到的数据切片和随机数状态都会变。新状态先写到 `state.tmp/`，
所有 rank 写完才替换旧状态；如果替换过程中断，`state.old/` 是恢复用的备份。不要为了重启删掉这些目录。

状态只在 `--save_state_every` 的整数倍步保存：间隔 100、共 5,123 步的训练，最后一个可续训状态通常是第 5,100 步，
而 `step05123/` 存的是最终推理权重。训练是否正常结束和最终 checkpoint 要分别确认，不要把 3 步冒烟测试的 checkpoint
当成完整训练的结果。

### GPU 上的优化核验（2026-10-14）

训练器仍是手写的 PyTorch / DDP / FSDP2。新增的优化：全参训练先在 meta 设备上建模型并切分、再由 rank 0 读 checkpoint
广播加载；24 槽位 KL / RPS 按 micro-batch 批量计算；下一步的分词在后台预取；完整的 DCP 训练状态。分词只是预取，
**不是**把单独分词的前缀拼接起来缓存，所以 prompt 的 token 边界、采样和选项置换都不变。可选的 shared prefix 和
DDP 通信重叠没有启用。

GPU 核验条件：4×H20，Qwen3.5-4B，seed 0，全局 batch 24，micro-token 预算 4,096，关闭 gradient checkpointing，
学习率表按 5,123 步，跑 3 步后停止。

| 检查项 | 改前 | 改后 | 说明 |
|---|---:|---:|---|
| LoRA 第 3 步 loss | 0.4320 | 0.4333 | 不是逐位一致 |
| LoRA 第 3 步验证 KL（100 行） | 0.450388 | 0.450648 | 差异很小但不为零 |
| LoRA 第 3 步 s/step | 1.99 | 2.04 | 这次冒烟测试没有体现提速 |
| 全参第 3 步 loss | 0.4264 | 0.4321 | 不是逐位一致 |
| 全参第 3 步验证 KL（100 行） | 0.449687 | 0.450884 | 未通过严格的数值等价检查 |
| 全参第 3 步 s/step | 2.35 | 1.65 | 仅冒烟测试计时 |
| 全参加载阶段的主机内存峰值（进程 RSS 之和） | 96.17 GiB | 31.01 GiB | 主机内存占用降低 |

3 步后 LoRA adapter / 决策头的最大参数差为 7.335e-6 / 3.943e-6。全参 bf16 checkpoint 的最大差为 0.00390625，
超过预定的 1e-3 门槛，这项严格检查仍未通过，不能宣称已通过。这个最大差只出现在
`layers.{0,21,25,26}.linear_attn.norm.weight` 各自的一个标量上；整个模型的 RMS 差为 1.205e-7，相对 L2 差为 1.021e-5，
4,205,751,296 个元素中有 415,674 个不同。这不是大范围参数错误的证据。RSS 是每秒采样一次进程树后求和，包含 import 和数据；
日志记录的"加载 + 切分"耗时改前 / 改后为 14 s / 16 s，所以内存降低不代表加载变快。另一个合成的决策头测试显示，TF32 会把
批量与逐行计算的最大 logit 差从 7.15e-7 放大到 6.85e-4，但这不能确定所有训练差异的原因。3 步也不能说明稳态吞吐或收敛是否等价。

**LoRA 续训检查完全通过**：跑 3 步、DCP 存档、新进程加载、再跑 3 步，得到的 adapter 和决策头与直接跑 6 步完全一致
（两者最大差都是 0，第 6 步验证 KL 都是 0.416699）。4B 的续训状态约 370 MB。这只是续训正确性检查，不是完整训练，
也不是对模型质量的评估。27B 实验使用已验证的 LoRA 路径，跨存档边界保留原有的 TF32 设置，没有使用尚未通过严格检查的全参路径。

> 去掉占位行会改变训练池，并可能让均匀 target 的验证行 KL 上升。做这个消融前先定好独立的开发 / 选择标准，
> 不要看了 test 结果再改 checkpoint 的选择方式。显式固定总步数，免得训练池变小后训练被悄悄缩短。

## 8. 显存与速度

参考训练：3×H20（96 GB），`--no_gc --micro_tokens 4096`，1.94 s/step，整体约 8.2k token/s，单卡显存峰值 48.8 GB，
5,123 步共 2.77 小时。显卡较小时保持 gradient checkpointing 开启（默认）和 / 或减小 `--micro_tokens`，
全局 batch 始终是 128 行。

CUDA 上请安装 `flash-linear-attention==0.5.2`，否则 transformers 会用很慢的参考实现跑 Gated DeltaNet 层。
`causal-conv1d` 可选。

## 9. 底座

已测试的 4B / 27B 底座都是 dense 的 `qwen3_5` 架构，决策用的 verbalizer token id 相同。`check_verbalizers` 会拒绝不兼容的
tokenizer。其他尺寸或架构需要另外检查初始化、LoRA 目标模块和推理；token id 兼容本身不代表训练效果好，也不代表可以即插即用。

| 底座 | 层数（全注意力） | hidden | 共享词嵌入 | LoRA 参数 | 状态 |
|---|---:|---:|:---:|---:|---|
| Qwen3.5-4B | 32（8） | 2560 | 是 | 30.5M | 已训练并评测（train-jev-4b-lora） |
| Qwen3.5-9B | 32（8） | 4096 | 否 | | 计划中 |
| **Qwen3.8-27B** | 64（16） | 5120 | 否 | 108.8M | 已训练并评测（train-jev-27b-lora） |

Qwen3.8-27B 说明：

- 架构尺寸和可训练参数量与 JEV-27B 的配方一致。这不能证明优化过程、训练数据选择或部署时的数值计算完全相同；
  对比时要说明协议。
- tokenizer 与 Qwen3.5 只差 7 个新增的音频 / TTS 特殊 token（id 248070–248076）；词表、合并规则和预分词完全相同，
  2,000 条训练 prompt 分出的 token id 完全一致。
- `AutoModelForCausalLM` 只加载文本模型（跳过视觉塔和 MTP 头），初始化检查 |Δp| = 0。`trainjev.eval_vision`
  会加载完整的 VL 模型来处理图像。
- 需要 `transformers>=5.17` 和 `peft>=0.21`；transformers 4.x 不认识 `qwen3_5` 架构。
- 已完成的 4×H20 训练：开启 gradient checkpointing，`--micro_tokens 16384`，单卡 74.9 GB，23.24 s/step，
  5,123 步含周期验证共 33.07 小时。启动脚本默认就是这 4 张卡的实测配置。

## 10. 调整配方

| 目的 | 改法 |
|---|---|
| 换底座尺寸 | `--base Qwen/Qwen3.5-9B` 或 `--base Qwen/Qwen3.8-27B`（代码不用改） |
| 加入自己的标注数据 | 转成上面的行格式，`--mix mydata.jsonl:2` |
| 增加容量 | `--lora_r 64`（α 自动设为 2r） |
| 只有硬标签 | target 用 one-hot；模型会更自信，记得重新拟合温度并关注 ECE |

## 11. v0.3.0 续训配方

发布模型不从头训练，而是在各自 v0.2.0 的 **LoRA + 决策头**上续训（`--init_adapter`，优化器重置）。
两尺寸使用相同的 692,953 行训练池：655,806 行旧蒸馏回放 + 37,147 行新监督。

> 本节是收到训练数据后的复刻指南，不是仅下载 HF 模型包即可执行的流程。加工后的数据、生成器、开发集和校准集未随 GitHub 或 HF 模型包公开；需要另外取得它们。不要把原始数据上传到模型仓库。

### 11.1 最小数据集合与哈希

复制数据时保留以下相对结构；下文 `$D` 指向移交的 `v030_20261017/` 目录。

| 文件 | 行数 | 用途 | SHA256 |
|---|---:|---|---|
| `base/train.jsonl` | 655,806 | 旧蒸馏回放 | `71e3ac5080e7da7d80d0dc867e3b4ac0307e1c3eb2c0843d578417f8510af9d6` |
| `math/train.jsonl` | 11,148 | 5,574 道 GSM8K，每题 near / broad 两个干扰项视图 | `f693a4cac3be41402c7a3d768db8578f1859af6394716539e923aac2fa35ed77` |
| `public_curated_train.jsonl` | 15,999 | PAWS 4,000、AQuA 3,999、QASC 4,000、HelpSteer2 4,000 | `8d44ca70d2acbf63e7cf66b4883fd9ef83485602f4dd2008834fe12c1279dcc2` |
| `rules/train.jsonl` | 10,000 | 六类独立校验的规则样本 | `f380ff3133f46bcdc378b016148b1df43cb0d6b116f5fb1409194f479c29df06` |
| `base/validation.jsonl` | 2,824 | 新任务训练内验证，不参与梯度更新 | `e8950a5019eade64a073646ddba72c07644c379d25ff9feee7bbbae5a7c24523` |

最小集合约 473 MiB，另需两个 v0.2.0 LoRA + head 起点、当前训练代码及发布包的 `training_recipe.json`。
原 `base/train.jsonl` 是绝对路径软链接，移交时用 `cp -L` / `rsync -L` 复制实体，避免断链。
**不要用原 jev-distill 的 validation 替换这里的 `base/validation.jsonl`**：训练器以 seed=1 固定抽取其中 2,500 行做验证。

### 11.2 来源与构建方法

精确重建需要固定上游版本，不应下载浮动 main 后重新随机划分。

| 来源 | 固定 revision | 用法 |
|---|---|---|
| `SargeDev/jev-distill-corpus-v3` | `75111a13a8455032f0435050bfc7e0c9026ecfe8` | 旧蒸馏 `train.jsonl`，核对上表哈希 |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | 同一批原题构建 near / broad 四选一视图 |
| `google-research-datasets/paws` | `161ece9501cf0a11f3e48bd356eaa82de46d6a09` | 释义判断，`noul` |
| `deepmind/aqua_rat` | `33301c6a050c96af81f63cad5562cb5363e88971` | 原生数学选择题 |
| `allenai/qasc` | `a34ba204eb9a33b919c10cc08f4f1c8dae5ec070` | 原生科学选择题 |
| `nvidia/HelpSteer2` | `990b2711a36180dd19d9c94b8627844866f8982a` | correctness 0–4 转为五选一 `choice`，不是 `score` |

- 公共来源只用上游 train split；seed=20261017，按连通 group 哈希分为 85/5/5/5%。先去重并排除旧输入重叠，再整组划分，不能逐行随机 split。
- `public/train.jsonl` 原有 16,000 行；组合构建时隔离了一个 AQuA group，实际训练必须用 **`public_curated_train.jsonl` 的 15,999 行**。
- GSM8K 经算术一致性、歧义和输入重叠筛选后保留 5,574 道训练原题；near / broad 按原顺序交错，不是 11,148 道独立原题。输入不附参考解析或正确标签。
- 六类规则为路由 1,630、显式证据 1,682、指代 1,646、退款换算 1,676、日历 1,676、条件计数 1,690 行。反事实对按 group 隔离，两个独立 state-only oracle 校验；trace 只作旁路证据。
- SNLI 未用于新增训练池；没有新增 memory-write 专项数据。QASC / HelpSteer2 保留 CC-BY-4.0 署名，PAWS 遵循其上游宽松许可。

完整构建链及原始资产需另外移交：`gold_pilot_20261016/`（`build_data.py`、`robust_eval.py`）、
`gold_balanced_20261016/`（`build_balanced.py`），以及 `$D` 中的 `public/builder.py`、`rules/build_rules.py`、
`build_combinations.py`、`build_suite.py`、raw、lock、manifest、许可与审计证据。
从原始 GSM8K 重建还需当时用于重叠排除的历史 DI 请求文件，不能用新版 DI37 请求替代。

移植前的核心生成器指纹：

| 文件（相对 `$D`） | SHA256 |
|---|---|
| `build_combinations.py` | `695e1ba75fec960560f40d83df767e2d83380bc7a9bef756bf77a6ec57ac84b6` |
| `public/builder.py` | `014ceb9063f30414a70573aad6df74b0475ed07b06b2a64382cced994a18d53f` |
| `rules/build_rules.py` | `d2751732e09f9115643e3bf98eec195e7166a78751211fe331f34aefb26784e6` |

生成器不在本仓库。只在新副本中修改绝对路径并重建，不覆盖封存数据；路径移植会改变代码哈希，但处理后 JSONL 必须与上表一致。
若缺少构建输入，优先使用已移交、哈希匹配的处理后文件，不宣称从原始来源精确重建。

### 11.3 实际采样

每步 global batch=128，题型固定为 `choice:64,noul:32,score:32`。
从有新增数据的题型中选 16 个位置做同题型替换，剩下 112 行为旧回放；新池三个文件各复制一次，题型内按池采样，不把来源均分。
替换保留原位置的损失权重。最终比例为 **0.125**，不是早期实验的 0.25。

seed=17，选项置换概率 0.3，placeholder 权重 0.05，损失 KL + 0.5×RPS，Brier 系数 0。
这不是按 epoch 遍历；到第 512 步共抽样 65,536 次（旧回放 57,344、新监督 8,192），不是唯一训练样本数。
抽样依赖文件顺序，选项置换还依赖 rank；原训练均为三卡，换 world size 不能保证相同训练轨迹。

### 11.4 两尺寸启动命令

```bash
# 在仓库内执行；D 换成移交数据的真实目录，输出目录必须是新的
D=/path/to/v030_20261017
REL=releases/v0.2.0
hf download shibing624/train-jev-4b-lora --revision v0.2.0 --local-dir "$REL/train-jev-4b-lora"
hf download shibing624/train-jev-27b-lora --revision v0.2.0 --local-dir "$REL/train-jev-27b-lora"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

# 4B
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m trainjev.train \
  --base Qwen/Qwen3.5-4B --data "$D/base" \
  --init_adapter "$REL/train-jev-4b-lora" \
  --mix "$D/math/train.jsonl,$D/public_curated_train.jsonl,$D/rules/train.jsonl" --mix_fraction 0.125 \
  --out runs/jev-4b-v030 --steps 2048 --batch 128 --micro_tokens 4096 --max_len 2048 \
  --kind_counts choice:64,noul:32,score:32 --lora_r 16 --lr 2e-5 --head_lr 4e-5 \
  --seed 17 --eval_every 256 --val_n 2500 --save_state_every 256 --no_gc \
  --rps_w 0.5 --brier_w 0 --perm 0.3 --placeholder_w 0.05

# 27B，与上面的 4B 顺序运行，避免争用同三张卡
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m trainjev.train \
  --base Qwen/Qwen3.8-27B --data "$D/base" \
  --init_adapter "$REL/train-jev-27b-lora" \
  --mix "$D/math/train.jsonl,$D/public_curated_train.jsonl,$D/rules/train.jsonl" --mix_fraction 0.125 \
  --out runs/jev-27b-v030 --steps 2048 --max_steps 1024 --batch 128 --micro_tokens 16384 --max_len 2048 \
  --kind_counts choice:64,noul:32,score:32 --lora_r 16 --lr 2e-5 --head_lr 4e-5 \
  --seed 17 --eval_every 256 --val_n 2500 --save_state_every 128 \
  --rps_w 0.5 --brier_w 0 --perm 0.3 --placeholder_w 0.05
```

两者发布点都是 **`step00512/`**，不是最后一步或训练器按最低验证 KL 保存的 `best/`。
只复刻已选点时，两者均可用 `--steps 2048 --max_steps 512`；**不能改成 `--steps 512`**，否则学习率曲线会变化。
完整训练实测：4B 三张 H20，2,048 步 1.3 h；27B 三张 H20，开启 gradient checkpointing，1,024 步 7.9 h。

### 11.5 开发集、校准与最终评测隔离

| 用途 | 另外需要的数据 | 数量 |
|---|---|---:|
| 训练内验证 | `base/validation.jsonl` | 2,824 行，固定抽取 2,500 |
| 发布点选择 | `suite.json` 的 `dev`、`audited/dev_*.jsonl` | 3,550，含 13 条辅助保真题 |
| 独立温度校准 | `suite.json` 的 `calibration`、`audited/calibration_*.jsonl` | 2,468 |
| 封存 PK | `suite_final2.json`、`reserve2/` | 2,494 金标题 |
| 外部 DI37 | 冻结的公开文本请求 | 110,087，只用于最后评测 |

校准集由 public 794、rules 800、GSM8K 512、legacy 362 行组成，不能用旧蒸馏 calibration 或 final2 代替。
完整复现还需移交 `evaluate_campaign.py`、suite 文件和选点记录；它们的路径在新副本中替换，题目、标签和行顺序不变。
原准则是开发集宏平均增益最高且 >16 选项视图不退步，两尺寸都选中第 512 步；不得用已知 final2 或 DI 成绩重新挑点。
温度只在独立 calibration 上拟合，写入新训练 checkpoint，不修改已发布模型包。

参考软件环境见 [requirements-tested.txt](../requirements-tested.txt)；数据构建另用 pyarrow 26.0.0、tokenizers 0.23.3。
数据、配方和种子一致不等于跨硬件或 kernel 逐位一致；重训后仍需检查打包前后概率和 manifest，不能提前保证权重 SHA 相同。
