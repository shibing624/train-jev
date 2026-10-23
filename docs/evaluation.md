# 评测

[English](https://github.com/shibing624/train-jev/docs/evaluation.en.md)

发布的 4B / 27B 模型用五套互补的评测衡量。只看训练集上的准确率不够；也不要把新的完整面板和旧的抽样结果混在一起比，
或者悄悄换掉读法 / 推理后端。

| 评测 | 衡量什么 | 规模 |
|---|---|---:|
| jev-distill test_set_30k / OOD | 与给定目标分布的一致程度 | 6,000 / 3,000 |
| [Decision Index 0.3 抽样](https://github.com/apolinario/decision-index) | 多领域的决策能力 | 6,000 题，35 个计分基准 |
| [Kev transfer-v4](https://github.com/jaredpalmer/kev) | Kev 训练时没见过的来源上的准确率和校准 | 656 题 |
| [看图决策](https://github.com/apolinario/decision-index) | 零样本的图像决策能力 | 5 个数据集共 1,444 题 |
| [train-jev-classification-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-classification-pk)（13 数据集分类 PK） | 中英文多数据集上的配对准确率 / 校准 | 1,300 题，13×100 |

超过 16 个选项的选择题另做一组对比：同一个 4B checkpoint，CLINC150 / BANKING77 / MASSIVE en 各 600 题。
全部结果和局限见 [results.md](https://github.com/shibing624/train-jev/docs/results.md)。所有评测都不用 test 标签拟合温度或挑选 prompt。

v0.3.0 另有 [train-jev-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-pk)：2,494 题、22 个任务，对应历史上称为“封存同题 PK”的结果。
这套评测含训练中出现过的公开来源和同族规则题，不是完全跨来源的泛化评测；固定题目不应再用于训练、选点或拟合温度。

## 1. 蒸馏语料与温度校准

用 `scripts/prepare_data.sh` 下载固定版本的语料和底座。评测样本用 Python `random.Random(7)` 抽取；
训练时的验证集抽样用 seed 1，与评测不同。target 是给定的概率向量，KL 不等于人工标注的准确率。
Top-1 / ECE 排除 target 并列最大的题，KL 统计全部题，所以分母不同。

```bash
# 自己训练的 checkpoint：在 calibration 切分上拟合温度（不用 test），温度存到 adapter 旁边
CKPT=runs/jev-4b/best FIT_CALIBRATION=1 OUT=eval/corpus-4b bash scripts/run_eval.sh
BASE=Qwen/Qwen3.8-27B CKPT=runs/jev-27b/best FIT_CALIBRATION=1 OUT=eval/corpus-27b bash scripts/run_eval.sh

# 已校准的发布模型：直接用它自带的温度，不要在 test 上重新拟合
bash scripts/run_eval.sh
python -m trainjev.evaluate --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --adapter_revision v0.2.0 \
  --data data/jev-distill-corpus-v3 --out eval/release-4b
python scripts/compare_corpus.py release=eval/release-4b/summary.json
```

`run_eval.sh` 默认用 `Qwen/Qwen3.5-4B` 和 `shibing624/train-jev-4b-lora`，应用发布模型的温度而不拟合。
只有显式设置 `FIT_CALIBRATION=1` 才传入 `--fit_calibration`；它会排除占位行，并把温度写进本地 checkpoint。
不能对 HF 缓存里的模型重新拟合；需要的话先下载到可写目录。用 `ADAPTER_REVISION` 固定发布模型版本。
历史上 4B 语料评测的 batch 是 24,576 token，27B 是 8,192；要完全复现推理实现，`--batch_tokens` 也要对上。
bf16 合并和批量推理可能带来很小的数值差异。

分布内的切分与训练集共享 state，但审计没有发现完整 prompt 重复；这些分数不能说明对新 state 的泛化。OOD 的 state 与训练集不重合。
新配方需要单独的、state / 分组（最好来源也）不重合的开发集，不要对着最终的 OOD / test 分数调参。

## 2. 推理服务的设置

```bash
BASE=Qwen/Qwen3.5-4B CKPT=shibing624/train-jev-4b-lora PORT=8020 STRATEGY=tournament bash scripts/run_serve.sh
```

评测走 `POST /v1/systemone`；本地后端只处理文本。读法策略、batch token 预算、最大长度、精度、温度和模型版本都要记录下来。
train-jev-4b-lora / train-jev-27b-lora 的历史宽选项结果用 `tournament`，[autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) 的参照用 `single`。
服务默认是 `single`，所以用默认设置复现不了历史的宽选项分数。要接 JEV 原生的 `/v1/decide` 后端，请用文档里的兼容代理，
不要把评分用的元数据当成 state 发过去。

## 3. Decision Index：复现原有抽样

用 [apolinario/decision-index](https://github.com/apolinario/decision-index) 的固定 revision `f8552a13b643d39f61185f5951bb5fed16bedf51` 作为抽样器和评分器：

```bash
git clone https://github.com/apolinario/decision-index third_party/decision-index
git -C third_party/decision-index checkout f8552a13b643d39f61185f5951bb5fed16bedf51
pip install -e third_party/decision-index
```

按该项目锁定的数据版本和文档重建 / 导入官方 0.3 suite。下载量可能很大或需要申请权限，不完整的文件不是有效的 suite。
用上游下载器，或者自行下载固定版本的文件并对照官方哈希校验后再导入。HLE / HoVer **不是我们 35 基准主结果的必需项**；
完整的 37 基准 suite 显式去掉它们，也能得到完全相同的原有抽样。

```bash
python scripts/di_sample.py --di_root third_party/decision-index --n 6000 \
  --drop 45,61 --seed 20260919 --rows_out data/di_sample6000.jsonl.gz \
  --suite_out third_party/decision-index/suite-0.3-sample6000
URL=http://127.0.0.1:8020 NAME=jev-4b bash scripts/run_decision_index.sh
```

完整 suite 放在别处时用 `--source_suite PATH` 指定。已有的输出不会被覆盖，请用新的带版本号的路径。
抽样器在符合条件的 catalog 之间轮流抽取整组，按 `seed:catalog:group` 的哈希排序。6,000 个输入请求覆盖 41 个 catalog；
公开指数只用去掉 45（HLE）/ 61（HoVer）后的 35 个计分基准，而不是 41 个都计入。

`sampling_manifest.json` 记录种子、各 catalog 的数量、有序 run ID 的 SHA256 和规范化模型输入的 SHA256。
已实际验证：用这些设置从完整的 37 基准 suite 重建，**全部 6,000 个有序 run ID 和有序的 state / questions 内容都与原抽样完全相同**。
gzip 头部 / 容器字节不作为是否相同的判据。

要对生成的**只含抽样的 suite** 评分，而不是完整 suite，否则没抽到的请求会被算成缺失。不要把这个调整过的 35 基准指数
说成官方完整榜单。`scripts/di_index.py` 读取官方的领域划分并去掉 45 / 61；新的 37 基准抽样或全量结果需要另外声明协议，
并重新跑参照模型。不要用猜的分数填补缺失的基准，也不要悄悄忽略失败的请求。

## 4. Kev transfer-v4

所有模型都用同一份冻结的 `transfer-v4` suite 和来源 revision，并核对 ID 和输入哈希。Kev 的实现和 suite 按
[jaredpalmer/kev](https://github.com/jaredpalmer/kev) 的说明另行获取。

```bash
# 在固定版本、带原始 suite 的 Kev 仓库里运行：
python -m kev.benchmark --remote http://127.0.0.1:8020 --remote-concurrency 1 \
  --suite evals/v4/transfer-v4 --out eval/transfer-jev-4b
```

准确率、NLL、ECE 只统计 656 道干净题，与对照 / 变体题分开报告（记录下来的预测元数据共 764 行）。这里的 `rows.json`
只有预测和标签，没有完整输入文本，不能用来证明没有污染。标签是每行选项 key 的下标，不一定是字母。
配对比较要求 ID / 题目 / 标签集合一致；逐题的 McNemar 显著性不能代替训练种子或来源层面的方差。

## 5. 看图决策

```bash
python -m trainjev.eval_vision --base Qwen/Qwen3.5-4B --adapter runs/jev-4b/best \
  --di_root third_party/decision-index --per_set 300 --out eval/vision-4b
python -m trainjev.eval_vision --base Qwen/Qwen3.8-27B --adapter runs/jev-27b/best \
  --di_root third_party/decision-index --per_set 300 --out eval/vision-27b
```

[Decision Index 上游工具](https://github.com/apolinario/decision-index) 需要先重建 5 个视觉数据集。实际覆盖：CV-Bench / BLINK / RealWorldQA / CharXiv 各 300 题，InfoVQA 244 题，
共 1,444 题而不是 1,500 题。mean skill / accuracy 是各数据集指标的等权平均，不是合并后的总准确率。
对比不同实现前，图片处理 / 尺寸、精度和题目 ID 都要对齐。文本 LoRA 合并进 VL 模型的语言部分，视觉塔没有训练。
这不是文本 HTTP 接口，也不是完整的多模态 HLE。官方 Decision Index 0.3 的文本适配器保留的是 501 道 HLE 文本选择题，
不是全部 2,500 道原始 HLE 题。

## 6. train-jev-pk、train-jev-classification-pk 与宽选项

两套固定题库均发布完整索引，但不是全部原文：[train-jev-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-pk) 有 2,494 题、22 个任务，
附带 1,594 题输入、其余 900 题仅含索引和标签；[train-jev-classification-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-classification-pk) 有 1,300 题、13 个任务，
附带 500 题输入、其余 800 题仅含索引和标签。未核验许可的原文不再分发；已包含的原文也保留上游署名和许可，
详情见各数据集 README，仓库代码许可不覆盖上游原文。

先按来源条款用 builder 获取原始来源文件，再离线恢复固定行号、校验标签和完整模型输入 SHA256：

```bash
python scripts/pk_build.py --out data/pk
python scripts/prepare_eval_data.py --dataset data/train-jev-pk --source-dir data/pk \
  --out eval/complete-inputs.jsonl
python scripts/evaluate_pk.py --dataset data/train-jev-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/train-jev-pk-4b

python scripts/prepare_eval_data.py --dataset data/train-jev-classification-pk --source-dir data/pk \
  --out eval/classification-complete-inputs.jsonl
python scripts/evaluate_pk.py --dataset data/train-jev-classification-pk --source-dir data/pk \
  --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --out eval/train-jev-classification-pk-4b
```

`prepare_eval_data.py` 不联网；缺失来源文件、标签不匹配或输入哈希不一致时拒绝输出完整题库。
`evaluate_pk.py` 同样要求恢复全部输入，使用本地 HF 原始 prefill，保留发布题库的 question / 等级说明，
不走会改写评分题说明的 HTTP 转换，也不重新拟合温度。train-jev-pk 按任务等权计算宏平均；历史分类 PK 的 ECE 使用报告中的 `all_rows.ece10`（合并后的 10-bin ECE），不能用各任务 ECE 的平均数替代。不要把已附原文的子集当成整套评测。
HF 模型加载仍可能下载模型；不同 GPU、batch、精度或推理栈不保证概率逐位一致，也不能据此改写历史分数。

旧的 `pk_benchmark.py` 仍可用于 builder 生成的每来源最多 600 行文件及服务对比。下面保留历史 13×100 分类服务协议与宽选项 builder：

```bash
python scripts/pk_benchmark.py run --url http://127.0.0.1:8020 --suite data/pk \
  --strategy tournament --limit 100 --workers 1 --out eval/pk_4b.jsonl
python scripts/pk_benchmark.py report model=eval/pk_4b.jsonl

python scripts/wide_build.py --out data/wide
URL=http://127.0.0.1:8020 NAME=jev-4b SUITE=data/wide LIMIT=600 bash scripts/run_wide.sh
```

对比前先冻结数据文件、标签和 SHA。历史分类 PK 用标签名作为选项（下划线换成空格），文本截断到 4,000 个字符，并使用记录下来的统一题目文字。
5 级评分任务按"在 5 个标签之间做选择"来问，而不是训练用的 6 级 `score` 协议。只比较场景 / 下标 / 标准答案都配对的结果，
不要拿不相关的平均数比。

宽选项评测记录了每次请求实际用的策略、选项顺序哈希和模型输入长度；9 个预热请求不计入。permute 带来的准确率提升和
置信度偏低见 [results.md §8](https://github.com/shibing624/train-jev/docs/results.md#8-超过-16-个选项的选择题)。读法策略和平均后的温度要在独立的开发集上选，再单独评测，
不要在这 600 道 test 题上拟合温度。

## 7. 不做的声明

- 固定的配方和输入可以复现；跨 GPU / 库版本 / 精度 / 合并方式的实现不保证逐位一致。
- 主结果用 HF 合并后的 bf16 推理。实验性的 vLLM 后端没有通过概率一致性门槛，它的结果不能替换报告里的概率。
- 单个训练种子和抽样指数不能证明在所有部署场景或完整 / 私有榜单上更优。
- 公开题库仅附带许可已核验的部分输入，其余行需要按上游条款取得来源文件恢复；不能声称打包了全部原文。私有日志不包含在仓库中，模型另从 HF 获取。
