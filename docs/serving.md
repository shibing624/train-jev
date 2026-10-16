# 部署与推理

[English](serving.en.md)

## 启动服务

```bash
# 从 Hugging Face 加载已发布的模型（自动下载底座、LoRA、决策头和温度）
python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --port 8020
python -m trainjev.serve --base Qwen/Qwen3.8-27B --adapter shibing624/train-jev-27b-lora --port 8020

# 自己训练的 checkpoint
CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--base` | `Qwen/Qwen3.5-4B` | 底座：HF id 或本地目录 |
| `--adapter` | – | HF repo id 或本地 checkpoint 目录（LoRA + `head.pt` + `calibration.json`） |
| `--adapter_revision` | – | adapter 的 HF tag 或 commit，例如 `v0.2.0` |
| `--strategy` | `single` | 超过 16 个选项的选择题怎么读（见下文） |
| `--device` | `cuda` | `cpu` 表示在 CPU 上用 fp32 推理（很慢，只适合核对） |
| `--host` / `--port` | `127.0.0.1` / `8020` | 监听地址 |
| `--max_len` | 32768 | prompt 的 token 上限，state 过长时从中间截断 |
| `--batch_tokens` | 32768 | 批量处理多道题时每次前向的 padded token 数 |
| `--calibration` | `<adapter>/calibration.json` | 温度文件 |
| `--base_only` | 关 | 用未训练的底座（决策头从 `lm_head` 初始化）作为参照 |
| `--proxy URL` | – | 把 JEV-27B-VL 的 `serve_decide.py` 服务挂到同一个 API 后面，方便对比 |

显存：4B 模型 bf16 约 9 GB，24 GB 显卡即可；27B 模型约 54 GB，需要 80 GB 显卡。CUDA 上请安装
`flash-linear-attention`，否则 Qwen3.5 的 Gated DeltaNet 层会走很慢的参考实现。

直接用 PEFT / Transformers 加载只会应用 LoRA adapter；决策头（`head.pt`）和温度（`calibration.json`）由 train-jev 应用，
所以请通过 `trainjev.serve` 提供服务。

## API

`POST /v1/systemone`，兼容 TypeSafe 的 [System One API](https://docs.typesafe.ai/api)。`GET /health` 返回模型名。

```jsonc
{
  "state": "…",                        // 字符串，或对象 / 数组（序列化成 JSON）
  "questions": {
    "<id>": {                          // id 由你自己定，模型看不到
      "type": "noul" | "choice" | "score",
      "instructions": "…",
      "criteria": …                    // noul：可选的 {true, false} 描述
                                       // choice：{选项: 描述 | null}，2–256 个选项
                                       // score：[等级, …]，从低到高排列
    }
  },
  "strategy": "single"                 // 可选，覆盖本次请求的 --strategy
}
```

| 类型 | 返回 |
|---|---|
| `noul` | `noul`：为 true 的概率 |
| `choice` | `choice`：概率最高的选项；`probabilities`：所有选项的概率 |
| `score` | `score`：期望等级（从 0 开始）；`probabilities`：各等级的概率 |

一个请求可以带任意多道题。它们共享同一段 state，但各自独立作答：每道题只看到 state 和自己的题目，看不到其他题的答案。
多道题会按长度分批推理。

6 个等级的 `score` 题使用训练过的 `score` 槽位，等级文字作为说明附在题目里；其他等级数的题会按"在等级文字之间做选择"
来问，`score` 返回期望下标。

`state` 里的图片只有 `--proxy` 模式支持。本地的看图决策用 `trainjev.eval_vision`（见 [evaluation.md](evaluation.md#5-看图决策)）。

## 示例

服务启动后：

```bash
python examples/client.py --url http://127.0.0.1:8020             # 一张工单，三种题型
python examples/batch_decisions.py --url http://127.0.0.1:8020    # 同一段 state 上的多个决策
python examples/smoke_systemone.py --url http://127.0.0.1:8020    # 三道答案已知的固定题
python examples/wide_choices.py --url http://127.0.0.1:8020 --compare   # 20 个选项，对比三种读法
```

不起服务，直接在 Python 里做一次决策：

```bash
python examples/local_inference.py --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora \
  --state "My card was charged twice for one order." --question "Which team should handle this request?" \
  --options billing shipping "technical support"
```

## 超过 16 个选项的选择题

决策头只训练过 A–P 这 16 个标签。17–256 个选项时由 `--strategy`（或请求里的 `"strategy"`）决定怎么读，
三种策略都来自 JEV-27B-VL 的 `serve_decide.py`：

| 策略 | 做法 | 前向次数（n 个选项） |
|---|---|---:|
| `single`（默认） | 一次前向：A–P 用训练过的决策头，Q–Z、AA、AB… 用底座自己的 `lm_head` 行，bias 为 0 | 1 |
| `tournament` | 每组 ≤16 个先分组打分，各组第一名加上最好的亚军凑成决赛；只读训练过的 A–P | ⌈n/16⌉ + 1 |
| `permute` | `single` 跑 4 种选项顺序（原顺序 + 3 种按题目文本固定种子的打乱）后取平均 | 4 |

在 train-jev-4b-lora 上实测，CLINC150（150 个选项）、BANKING77（77 个）、MASSIVE en（59 个）各 600 题：

| 策略 | 准确率（1,800 题） | ECE ↓ | p50 延迟 |
|---|---:|---:|---:|
| `single` | 0.727 | 0.162 | 52 ms |
| `tournament` | 0.732 | **0.050** | 109 ms |
| `permute` | **0.771** | 0.218 | 171 ms |

`permute` 最准但偏保守（置信度偏低）；`tournament` 校准最好。分数据集的结果见
[results.md](results.md#8-超过-16-个选项的选择题)。两个发布模型的公开评测分数都用 `tournament`，
要复现请加 `--strategy tournament`。
