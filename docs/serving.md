# 部署与推理

[English](https://github.com/shibing624/train-jev/docs/serving.en.md)

## 启动服务

```bash
# 脚本默认加载 Qwen/Qwen3.5-4B + shibing624/train-jev-4b-lora
bash scripts/run_serve.sh

# 从 Hugging Face 加载已发布的模型（自动下载底座、LoRA、决策头和温度）
python -m trainjev.serve --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora --port 8020
python -m trainjev.serve --base Qwen/Qwen3.8-27B --adapter shibing624/train-jev-27b-lora --port 8020

# 自己训练的 checkpoint
CKPT=runs/jev-4b/best PORT=8020 bash scripts/run_serve.sh

# 固定 adapter 版本，并通过尾随 CLI 参数调整 token 预算
ADAPTER_REVISION=v0.3.0 STRATEGY=tournament bash scripts/run_serve.sh --batch_tokens 32768

# 同时支持文本和图片（完整 VL 底座 + 原 LoRA/head/温度，不重新训练）
VISION=1 bash scripts/run_serve.sh
```

`run_serve.sh` 支持 `BASE`、`CKPT`、`ADAPTER_REVISION`、`DEVICE`、`NAME`、`HOST`、`PORT`、`STRATEGY`、`VISION`，
尾随参数直接传给 `trainjev.serve`。脚本默认 `BASE=Qwen/Qwen3.5-4B`、`CKPT=shibing624/train-jev-4b-lora`、
`DEVICE=cuda`、`NAME=train-jev-4b-lora`、`HOST=127.0.0.1`、`PORT=8020`、`STRATEGY=single`；adapter 版本默认不固定。
下表是 Python CLI 的默认值，`--adapter` 没有默认值，与脚本不同。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--base` | `Qwen/Qwen3.5-4B` | 底座：HF id 或本地目录 |
| `--adapter` | – | HF repo id 或本地 checkpoint 目录（LoRA + `head.pt` + `calibration.json`） |
| `--adapter_revision` | – | adapter 的 HF tag 或 commit，例如 `v0.3.0`；旧版用 `v0.2.0` |
| `--strategy` | `single` | 超过 16 个选项的选择题怎么读（见下文） |
| `--device` | `cuda` | `cpu` 表示在 CPU 上用 fp32 推理（很慢，只适合核对） |
| `--host` / `--port` | `127.0.0.1` / `8020` | 监听地址 |
| `--max_len` | 32768 | prompt 的 token 上限，state 过长时从中间截断 |
| `--batch_tokens` | 32768 | 批量处理多道题时每次前向的 padded token 数 |
| `--calibration` | `<adapter>/calibration.json` | 温度文件 |
| `--base_only` | 关 | 用未训练的底座（决策头从 `lm_head` 初始化）作为参照 |
| `--vision` | 关 | 加载完整 Qwen VL 底座，在同一接口接受图片 assets；额外占用视觉编码器显存 |
| `--proxy URL` | – | 把 [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) 的 `serve_decide.py` 服务挂到同一个 API 后面，方便对比 |

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

本地图片请求需要启动时加 `--vision`。不开启则保留原纯文本加载方式，图片 assets 返回 422，而不是当文本忽略。视觉模型与纯文本路径共享同一个已应用 LoRA 的语言模型和 FP32 决策头；图片题逐题运行，纯文本题仍按长度批处理。

## 图片输入：TypeSafe assets

采用 [TypeSafe Images 接口](https://docs.typesafe.ai/images) 的 assets 格式，支持 `state`、`instructions`、`criteria` 的字符串值内引用图片。本地服务不要求额外实验 header；启动时启用 `--vision` 即可。支持现有 `noul` / `choice` / `score`，**不支持 `box` 定位题**。

推荐 multipart：无需 base64 的约 33% 体积膨胀。JSON 必须是第一部分，名字 `request`；图片用不同的文件 part：

```bash
curl http://localhost:8020/v1/systemone \
  -F 'request={
    "model": "jev-latest",
    "state": {"door": "a3f1c2e8-door"},
    "assets": {"a3f1c2e8-door": {"type": "image", "part": "door-photo"}},
    "questions": {
      "person_present": {"type": "noul", "instructions": "Is there a person in `door`?"},
      "carrying": {"type": "choice", "instructions": "What is the person in `door` carrying?",
                   "criteria": {"package": "A box or parcel.", "bag": "A shopping bag.", "nothing": "Empty hands."}}
    }
  };type=application/json' \
  -F 'door-photo=@door-camera.jpg;type=image/jpeg'
```

不方便 multipart 时用 JSON/base64。无需把图片预先缩放到固定尺寸：

```python
import base64
import uuid
from pathlib import Path
import httpx

photo_key = str(uuid.uuid4())
photo = {
    "type": "image",
    "data_base64": base64.b64encode(Path("door-camera.jpg").read_bytes()).decode("ascii"),
}
response = httpx.post(
    "http://localhost:8020/v1/systemone",
    json={
        "model": "jev-latest",
        "state": {"door": photo_key},
        "assets": {photo_key: photo},
        "questions": {"person_present": {"type": "noul", "instructions": "Is there a person in `door`?"}},
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["answers"])
```

### 引用和校验

- 只替换 `state` / `instructions` / `criteria` 的字符串值，不替换 question ID 或 Choice 选项名等对象键。整值优先，长字符串从左到右匹配；引用可在句中出现，推荐 UUID，避免短词误匹配。
- 可以嵌套多图，或把例图放进单题 instructions / criteria。该题只看到自己的例图，其他题不会看到它；同一个引用每用一次都展开一次。
- 每个 asset 必须被上述字段引用。`type` 可省略；`data_base64` 使用标准带 padding 的 base64，不带 `data:` 前缀或空白；`format` 可省略，由文件内容自动识别，提供时须与内容一致。同时有 part 和 base64 时 part 优先。
- 支持 PNG、JPEG、WebP、BMP、TIFF、GIF、ICO、AVIF（具体解码能力取决于 Pillow）。GIF、多页 TIFF 和 APNG 取第 0 帧，作为静态图片；SVG、PDF、PostScript 和视频不在此接口范围内。
- 不再限制每边必须是 280–1568 像素。普通图片由 Qwen 图像处理器按模型的 pixel/grid 配置缩放；极小图片等比放大，超长宽比图片缩小并补边，不裁掉内容。EXIF 方向校正、转 RGB 和去元数据后送入视觉编码器。
- 保留资源保护：每文件 16 MiB、HTTP 请求体 32 MiB；处理后图片及展开请求各最多 256 MiB，总解码像素约 89.5 MP，最多 16 个独立 asset、64 次引用。保留 Pillow 解压炸弹检查，超出资源预算返回 413；这些是安全上限，不是 TypeSafe 的固定边长要求。
- 图片 prompt 超过 `--max_len` 返回 422，不静默截断图片或其文字。旧纯文本的中间截断行为不变。
- 非法图片/引用/JSON 返回 400，非法题型/选项或未启用视觉返回 422，不支持的 Content-Type / Content-Encoding 返回 415。本地目前只接受未压缩 JSON 或 multipart。
- 推理内存不足返回 503，不返回部分答案；减少图片数或上下文后再试。可确定超长的重复图片在分配 patch 张量前拒绝。
- assets 不接受 URL 或服务器本地文件路径，不自动访问外网或读任意文件。`--proxy` 会将已验证的图片转为 data URL 转发给配置的 `/v1/decide`；题目或标准中的图片会带标签放入该题的上游 state，避免丢图。原 proxy 的 inline-image state 格式仍可使用。

`GET /health` 的 `images` 字段说明是否开启本地视觉或 proxy。请求 `model` 是兼容字段，实际模型由启动参数确定，不按请求切换模型。该本地服务不验证 TypeSafe 的 Bearer key；公网部署请在反向代理配置鉴权、限流和总请求体限制。

这是图片输入能力，不保证每个视觉判断都正确；模型仍是仅文本训练后的零样本视觉迁移。可用 `trainjev.eval_vision` 验证图像任务（见 [evaluation.md](https://github.com/shibing624/train-jev/docs/evaluation.md#5-看图决策)）。

## 示例

服务启动后：

```bash
python examples/client.py --url http://127.0.0.1:8020             # 一张工单，三种题型
python examples/batch_decisions.py --url http://127.0.0.1:8020    # 同一段 state 上的多个决策
python examples/smoke_systemone.py --url http://127.0.0.1:8020    # 三道答案已知的固定题
python examples/wide_choices.py --url http://127.0.0.1:8020 --compare   # 20 个选项，对比三种读法
python examples/image_decisions.py --image door-camera.jpg           # TypeSafe multipart 图片
python examples/image_decisions.py --image door-camera.jpg --transport base64  # JSON/base64
```

不起服务，直接在 Python 里做一次决策：

```bash
python examples/local_inference.py --base Qwen/Qwen3.5-4B --adapter shibing624/train-jev-4b-lora \
  --state "My card was charged twice for one order." --question "Which team should handle this request?" \
  --options billing shipping "technical support"
```

## 超过 16 个选项的选择题

决策头只训练过 A–P 这 16 个标签。17–256 个选项时由 `--strategy`（或请求里的 `"strategy"`）决定怎么读，
三种策略都来自 [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) 的 `serve_decide.py`：

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
[results.md](https://github.com/shibing624/train-jev/docs/results.md#8-超过-16-个选项的选择题)。两个发布模型的历史分类 PK / 宽选项评测用 `tournament`，
要复现这些结果请加 `--strategy tournament`；完整公开 Decision Index 文本面板用 `single`，不能把不同协议混在一起。
固定发布题库 [train-jev-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-pk) / [train-jev-classification-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-classification-pk)
请按 [评测文档](https://github.com/shibing624/train-jev/docs/evaluation.md#6-train-jev-pktrain-jev-classification-pk-与宽选项)恢复缺失输入并用本地 HF 评测脚本运行，
保留原始问题与评分等级说明，不用 HTTP 改写代替该协议，也不保证所有 GPU / batch 推理栈的概率逐位一致。
