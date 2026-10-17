# qwen25vl-lite（中文说明）

一个**面向 8 GB 消费级显卡**（RTX 5060 / 4060 这一档）的小型、可通读的 Qwen2.5-VL 推理引擎。

它不是 vLLM 的替代品。它把"把视觉语言模型塞进一张小卡并跑出像样吞吐"真正需要的那几块**完整地、自包含地实现出来**，并且用一套和 HuggingFace `transformers` 逐项数值对照的测试去验证这些实现——而不是断言形状然后祈祷。

[English README](README.md)

---

## 目录

1. [支持什么模型](#一支持什么模型)
2. [部署](#二部署)
3. [使用](#三使用)
4. [加速原理](#四加速原理)
5. [性能数据](#五性能数据)
6. [验证了什么、没验证什么](#六验证了什么没验证什么)
7. [已知限制](#七已知限制)
8. [测试](#八测试)

---

## 一、支持什么模型

### 架构

**只支持 Qwen2.5-VL 一种架构。**

引擎里写死了这个架构的各个细节，没有 `model_type` 分派机制：

| 组件 | 位置 | 说明 |
|---|---|---|
| 配置解析 | `config.py` | 同时兼容旧版 `rope_scaling` 和 transformers 5 的 `rope_parameters` |
| MRoPE | `rope.py` | 文本用 MRoPE，视觉用二维 RoPE，`get_rope_index` |
| 视觉塔 | `vision.py` | **窗口注意力 + 全局注意力混合**：`window_size=112`，全局层为 `fullatt_block_indexes = [7, 15, 23, 31]`，末端 PatchMerger |
| 解码器 | `text.py` | GQA + SwiGLU + MRoPE，支持 paged / chunked / dense 三种注意力 |
| 图像预处理 | `imageproc.py` | `smart_resize` + patchify，与 HF processor 的 patch 顺序**逐位一致** |

因此换一个架构（Qwen2-VL、Qwen3-VL、或其他）会在加载权重或前向时直接失败——不存在"试试看"的路径。这是刻意的：宁可显式失败，也不要静默算错。

### 尺寸

**任何 Qwen2.5-VL 尺寸都可以**（3B / 7B / 32B / 72B），前提是按 `qwen25vl-lite plan` 规划后能装下：

- **3B 是 8 GB 卡的甜点**。本项目的性能数字都是在 3B 上测的；正确性声明则是**架构级**的——测试用现场构造的随机权重 tiny Qwen2.5-VL 与 HuggingFace 对照，不需要下载 checkpoint。
- 7B 及以上需要更大的卡，或者启用 `offload.py` 的**静态层放置**（部分层留 GPU，其余从 pinned host memory 流式加载，双缓冲预取）。

### 权重格式

- 输入是 **HuggingFace safetensors 快照目录**（含 `config.json`），即官方发布的原始权重。
- **不需要预先量化好的权重**。int8 / int4 / AWQ 都是**加载时运行时量化**的，所以你能直接用官方 checkpoint。

### 计算精度

- `--dtype bfloat16` 是**推荐值**，也是 checkpoint 自己声明的 `torch_dtype`。
- ⚠️ **`float16` 在大图上不安全**：视觉塔会在大的 patch grid 上溢出成 NaN（1312×1199 图在 `full` 档 = 1258 视觉 token 时产生 NaN 特征），于是所有 logit 都是 NaN，模型只会重复吐一个 token。HuggingFace `transformers` 在 fp16 下**表现完全一样**，所以这是 checkpoint 的精度极限，不是引擎 bug。**用 bfloat16。**

---

## 二、部署

### 2.1 安装

```bash
# 运行时依赖：torch + transformers（服务器需要 tokenizer）
pip install -e ".[hf]"

# 开发/测试：pytest + 固定版本的 transformers（等价性测试对版本敏感）
pip install -e ".[dev]"

# 可选：camera 子命令需要 OpenCV
pip install -e ".[camera]"
```

`qwen25vl-lite` 这个入口脚本**属于你 pip install 的那个环境**。如果你的 shell 激活在别的环境（比如裸 `base`），就会 `command not found`——先激活对应环境，或用全路径：

```bash
conda activate dl                                   # 或任何装了它的环境
qwen25vl-lite serve --help

/home/whu/miniconda3/envs/dl/bin/qwen25vl-lite serve --help
/home/whu/miniconda3/envs/dl/bin/python -m qwen25vl_lite.cli serve --help
```

### 2.2 先做显存规划（强烈建议）

```bash
qwen25vl-lite plan --model ./Qwen2.5-VL-3B-Instruct \
                   --vram-gb 8 --ram-gb 96 --weight-dtype int4 --context 32768
```

实际输出：

```
KV cost: 2 x 36 layers x 2 kv_heads x 128 head_dim x 2 B = 36.0 KiB/token

memory plan
  weights (int4 )        3.53 GiB
  KV per token             36.0 KiB
  GPU-resident KV pool     3.00 GiB (87,350 tokens, float16)
  vision/prefill reserve   95.00 MiB
  runtime + allocator      589.82 MiB
  ----------------------------------------
  total on GPU             7.20 GiB
  decoder layers on GPU    36 of 36
  max concurrent sequences 42
  prefill chunk size       3328
  note: host memory could hold roughly 2,690,615 more KV tokens (92.37 GiB); these are swap capacity, not GPU-resident capacity.
  note: logical context ceiling is 128,000 tokens from the config; the GPU-resident figure above is what is fast, and the host figure is what is merely possible.
```

**三个"上下文长度"必须分开看**，这是很多 README 不诚实的地方：

| 概念 | 含义 |
|---|---|
| 逻辑上下文 | 能接受的最长序列，受 `max_position_embeddings` 和换页能力限制 |
| GPU 常驻 KV | KV 真在显存里的 token，**这些才是全速解码的** |
| CPU 换页容量 | 能换到内存的额外 token，**解码更慢，必须实测** |

Qwen2.5-VL-3B 的 KV 成本是**算出来的**，不是拍的：

```
2 (K 和 V) × 36 层 × 2 个 kv head × 128 head_dim × 2 字节 = 36,864 字节 = 36 KiB/token
```

所以 250,000 token 的 fp16 KV 需要 **8.58 GiB——比整张卡还大**。任何声称 8 GB 上能跑 25 万 token 的方案，说的是换页容量，不是显存常驻容量。

### 2.3 启动服务

```bash
qwen25vl-lite serve \
  --model ./Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --weight-dtype int4 \
  --group-size 128 \
  --kv-dtype float16 \
  --num-blocks 4000 \
  --block-size 16 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --vision-tier full \
  --int4-kernel \
  --paged-attn-kernel \
  --cuda-graph \
  --host 127.0.0.1 \
  --port 8081
```

**这三个 `--*-kernel` / `--cuda-graph` 开关建议一起开**，原因见[加速原理](#四加速原理)。每个开关启动时都会自检，不合格会自己拒绝并回退到参考路径，并打印原因。

### 2.4 后台常驻

Python 后台运行时 stdout 不是 tty，会**块缓冲**——自检输出只在进程退出时才刷出来，日志平时是空的。加 `PYTHONUNBUFFERED=1`：

```bash
export PYTHONUNBUFFERED=1
setsid nohup qwen25vl-lite serve ... > serve.log 2>&1 < /dev/null &
disown
```

启动后日志应该是：

```
loaded 824 tensors (252 quantised) in 20.6s
--int4-kernel: enabled; worst relative |fused - reference| = 6.49e-03 over 4 layer(s)
--paged-attn-kernel: enabled; worst relative |fused - reference| = 3.50e-03
serving Qwen2.5-VL-3B-Instruct on http://127.0.0.1:8081  (Ctrl-C to stop)
```

注意 `--cuda-graph` 启动时**不打印任何东西**——它的自检发生在首次 decode 步捕获时（懒执行）。

### 2.5 关键参数一览

| 参数 | 取值 | 说明 |
|---|---|---|
| `--device` | `cpu` / `cuda` / `cuda:0` | |
| `--dtype` | `float32` / `float16` / `bfloat16` | **推荐 bfloat16**；fp16 在大图上会 NaN |
| `--weight-dtype` | `float32` / `float16` / `bfloat16` / `int8` / `int4` / `awq` | `int4` 是 8 GB 的关键 |
| `--group-size` | 默认 128 | int4 分组大小 |
| `--kv-dtype` | `float16` / `bfloat16` / `float32` / `int8` | int8 KV 可减半 |
| `--quantize-kv` | flag | 启用 KV 量化 |
| `--num-blocks` / `--block-size` | 默认 2048 / 16 | paged KV 的块数/块大小 |
| `--max-num-seqs` | 默认 8 | 最大并发序列 |
| `--max-num-batched-tokens` | 默认 4096 | 单次迭代的 token 上限 |
| `--max-model-len` | 默认 32768 | 逻辑上下文上限 |
| `--no-prefix-cache` | flag | 关闭 radix 前缀缓存 |
| `--no-chunked-prefill` | flag | 关闭分块 prefill |
| `--vision-tier` | `full` / `half` / `thumbnail` / `pruned` | ⚠️ 见下方说明 |
| `--fastv` / `--fastv-ratio` / `--fastv-layer` | | 显式启用 FastV 图像 token 剪枝 |
| `--preemption` | `swap` / `recompute` | 显存不足时的抢占策略 |
| `--attn-backend` | 如 `math,flash` | 钉住 SDPA 后端；默认运行时探测最快的可用后端 |
| `--int4-kernel` | flag | 融合 INT4 反量化 + GEMM |
| `--paged-attn-kernel` | flag | 融合 paged decode attention（flash-decoding） |
| `--cuda-graph` | flag | decode 步捕获成 CUDA graph |

> ⚠️ **`--vision-tier` 这个服务端 flag 不改图像像素预算。** 图像预算由**请求体**里的 `vision_tier` / `max_vision_tokens` 决定（默认 `full`）；服务端的 `--vision-tier` 只进 `EngineConfig.vision_tier`，只驱动 FastV 剪枝。实测：用 `--vision-tier thumbnail` 重启服务端，`prompt_tokens` 前后都是 421，毫无变化。**要降视觉 token，用请求体字段。**

---

## 三、使用

### 3.1 CLI 子命令

| 命令 | 作用 |
|---|---|
| `plan` | 打印显存规划和 KV 算术（三个上下文数字分开） |
| `probe` | 探测这台机器上哪些 SDPA 后端真的可用 |
| `tiers` | 列出视觉 token 策略 |
| `bench` | 跑一个短基准（吞吐 + 显存） |
| `generate` | 用 token ids 做一次性生成 |
| `serve` | 启动 OpenAI 兼容服务 |
| `monitor` | 轮询运行中服务的 `/stats`，实时显示解码速率 |
| `camera` | 抓摄像头帧并让模型实时描述 |

### 3.2 HTTP 接口

服务端只用**标准库**实现（无 FastAPI/uvicorn 依赖）。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/stats` | 调度器/缓存计数器 + 实时解码速率 |
| GET | `/v1/models` | 模型信息 + 可用的 attn 后端 + 视觉档位列表 |
| POST | `/v1/chat/completions` | OpenAI chat 格式，支持 `image_url` 内容块（base64 `data:` URL 或 http(s)） |
| POST | `/v1/completions` | 支持 `prompt_token_ids`（**不需要 tokenizer**） |

**请求体支持的字段**：`temperature`、`top_p`、`top_k`、`max_tokens`（或 `max_completion_tokens`）、`repetition_penalty`、`presence_penalty`、`frequency_penalty`、`seed`、`ignore_eos`、`stop`、`stream`，以及本项目特有的：

- `vision_tier`：`full` / `half` / `thumbnail` / `pruned`（**这才是图像预算的开关**）
- `max_vision_tokens`：直接指定视觉 token 上限，**覆盖** `vision_tier`
- `prompt_token_ids`：`/v1/completions` 用，绕过 tokenizer

### 3.3 文本对话

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen25vl-lite",
    "messages": [{"role": "user", "content": "用五句话解释什么是 MRoPE。"}],
    "max_tokens": 256, "temperature": 0
  }'
```

### 3.4 图像理解

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen25vl-lite\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG\"}},
      {\"type\": \"text\", \"text\": \"这张图里有什么？\"}
    ]}],
    \"max_tokens\": 128, \"temperature\": 0,
    \"max_vision_tokens\": 128
  }"
```

Python 客户端：

```python
import base64, json, urllib.request

url = "data:image/jpeg;base64," + base64.b64encode(open("photo.jpg", "rb").read()).decode()
body = {
    "model": "qwen25vl-lite",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "描述这张图。"},
    ]}],
    "max_tokens": 128, "temperature": 0,
    "max_vision_tokens": 128,      # 降视觉 token 最直接的开关
}
req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

### 3.5 实时监控（monitor）

```bash
qwen25vl-lite monitor --url http://127.0.0.1:8081                 # 默认 0.5 s
qwen25vl-lite monitor --url http://127.0.0.1:8081 --interval 0.1  # 10 Hz
qwen25vl-lite monitor --url http://127.0.0.1:8081 --rows          # 每样本一行，便于复制/记日志
qwen25vl-lite monitor --url http://127.0.0.1:8081 --json          # 每样本一个 JSON
```

tty 下它**覆盖写同一行**（所以你复制终端内容时会看到所有样本被压成一行，这是正常的）。`--rows` 或非 tty 时每样本一行。

```
time     decode tok/s  step ms  run  wait  swap     kv blocks prefix hit   idle gpu                    history
07:58:58         77.6    11.68    1     0     0      9/4000         0.0%      — gpu  98%    5.9/8.0 GiB ▁█
07:59:00         60.6    13.22    1     0     0     17/4000         0.0%      — gpu  98%    6.0/8.0 GiB ▂▆█▆▄▆█▆▁
07:59:01         75.1    13.17    0     0     0      2/4000         0.0%      — gpu  98%    6.0/8.0 GiB ▂▆█▆▄▆█▆▁▃▇▅
07:59:01          0.0     0.00    0     0     0      2/4000         0.0%   0.5s gpu  97%    6.0/8.0 GiB ▇███▇███▆▇█▇▇▁
07:59:03          0.0     0.00    0     0     0      2/4000         0.0%   2.1s gpu   0%    6.0/8.0 GiB ▇███▇███▆▇█▇▇▁▁▁▁▁▁▁
```

速率来自服务端最近 64 个 decode 步的滚动窗口，并且**只统计足够新的步**——所以空闲时读数是 `0.0`，`idle` 列显示安静了多久，而不是抱着最后一次的速率不放。

轮询很便宜：`/stats` 往返空闲时 **0.43 ms**、负载下中位 **1.23 ms**，所以 10 Hz 只花约 1% 的一个核。`nvidia-smi` 一次约 **16 ms**（是 `/stats` 的约 40 倍），所以它按自己的 `--gpu-interval`（默认 1 s）刷新并复用。

### 3.6 摄像头实时描述（camera）

```bash
qwen25vl-lite camera --url http://127.0.0.1:8081 --device 0
qwen25vl-lite camera --url http://127.0.0.1:8081 --device 0 --width 320        # 推荐：快约 40%
qwen25vl-lite camera --url http://127.0.0.1:8081 --max-vision-tokens 128       # 等价效果
qwen25vl-lite camera --url http://127.0.0.1:8081 --quiet                       # 只打印描述
```

它从 V4L2 设备抓帧，把每帧 POST 给**正在运行的服务**——复用已加载的模型，而不是再开一份（服务中的模型已占 8 GB 卡的约 5.9 GB，再开一份会 OOM）。

```
[08:16:44]    521 ms (1.92 fps)  A bed with a computer monitor on a table in the background.
[08:16:44]    504 ms (1.99 fps)  A cluttered room with a bed, desk, and computer monitor.
[08:16:45]    516 ms (1.94 fps)  A cluttered room with a bed, desk, and computer monitor.
```

**视觉 token 是每帧耗时的大头，有两个杠杆**（同一帧实测，交错测量，归一化到固定 16 token 答案）：

| 设置 | 视觉 token | prefill | fps |
|---|---|---|---|
| 原生 640，`full`（默认） | 421 | 251 ms | 2.31 |
| **`--width 320`**（客户端降采样） | 129 | 133 ms | **3.20**（+38.5%） |
| **`--max-vision-tokens 128`**（请求字段） | 147 | 138 ms | **3.23**（+40.1%） |
| `--vision-tier thumbnail`（请求字段） | 264 | 184 ms | 2.66（+15.3%） |

描述质量在各档位都站得住——同一帧在 421 token 下是 "A person is laying on a bed with a phone and a pillow"，在 129 token 下是 "A person is lying on a bed with a phone in their hand"，没有退化成乱码。低于约 130 视觉 token 后 prefill 会碰到 **~130 ms 地板**（文本 prompt + 固定请求开销），再降没有收益。

实拍对照：原生 640 → **1.79 fps**，`--width 320` → **~2.55 fps**（+42%）。

---

## 四、加速原理

### 4.1 前提：先把瓶颈定对

所有加速都建立在一个诊断上：**一个 decode 步不是算力受限，而是发射受限（launch-bound）**。

RTX 5060 + Qwen2.5-VL-3B INT4/group-128、512 token context 下实测：

- CPU 把一个 decode 步**排进队列**需要 **19.79 ms**
- 它背后真正的 GPU 工作量只有 **13.60 ms**

也就是说 GPU 每个 token 有约三分之一时间在空转等 CPU。原因是一步里有 **约 2100 次 Python 层算子派发**和 **252 次 Triton launch**。

方向由此确定：**要减少"发射"，而不是优化 kernel 的算术效率**。如果没测这一步，很容易一头扎进调 tile size / num_warps，而那是错的方向。

### 4.2 杠杆一：CUDA graph —— 消灭发射本身

把整个 decoder forward 捕获成一张 CUDA graph，之后一步就是**一次 launch**，那 2100 次派发和 252 次 launch 全部消失。

硬约束是**形状必须固定**。这里有个会咬人的细节：block table 的宽度每 16 个 token 就增长一格，形状一直在变。两个选择：

- 每变一次就重捕：**亏**。捕获要预热 forward + 验证 replay，约 **34 ms**，而 graph 只省约 **4.5 ms**。
- 把宽度**向上取整到桶**（默认 16 个 block = 256 token），表尾填 padding：捕获频率从每 16 token 一次降到每 256 token 一次，成本摊薄到每 token 不到 1 ms 的零头。

桶宽 16 是在 4/8/16/32/64/256 里扫出来的最优值。

**单独开的收益只在 CPU 是瓶颈的场景**：512 context 下 51.3 → 61.8 tok/s；但 2048 context 下 eager 本身已经是 GPU-bound，graph 自己的调度开销约 1 ms，反而 **47.9 → 45.4 变慢**。

一个真实的坑：捕获用的 buffer 必须**零填充而不是 -1 填充**——融合 paged attention kernel 会用读到的 block id 索引缓存，`-1` 会访问到张量之前的字节，在捕获时炸成 illegal memory access。

### 4.3 杠杆二：融合分页 attention —— 消灭"为了适配库而多做的功"

参考路径的做法是：把每层整个 context **gather 成稠密张量**，配上加性长度 mask，交给 SDPA。

在 decode 场景（每个序列只有 1 个 query）这个做法**和硬件严重错配**：

- 单层在 512 context 下要 50 us、2048 下要 183 us，而**内存带宽下限只有 5–22 us**——差一个数量级，说明在做大量无用功。
- 稠密 gather 本身还要额外 0.15 ms / 0.54 ms 每步。
- M=1 时 SDPA 各后端都是为 M 较大设计的，在 M=1 上效率极低。

融合 kernel **直接读分页缓存**：遍历 block table、在 `context_lens` 处停下、跑 online softmax。于是**稠密 gather 不存在了，mask 张量也不存在了**。它是 flash-decoding：context 切分到多个 program 并行，再用第二个小 kernel 合并 partial。

设计上的坑：`can_use()` **只允许检查静态元数据**。因为它会在 graph 捕获内部被调用，而 `context_lens.min()` 是 device→host 拷贝，在图捕获里非法。零长度 context 改在 kernel 内部用 `tl.maximum(..., 1)` 处理。

**单独开它是在拿 GPU 时间换 CPU 时间**：512 context 下 52.1 → 50.5 **变慢**（每层两次 Triton launch 的入队成本超过它替代掉的那一次 SDPA）；2048 下 47.9 → 49.7 才转正。

### 4.4 杠杆三：INT4 —— 消灭 92% 的无用搬运 + 消灭 16× 的无用计算

**(a) 参考路径每步都在"解包"权重。** 它把每个量化权重**物化成稠密的 fp16 副本**，每个 decode 步都做一遍。实测**这个解包占整个 decode 步的约 92%**：一个 2048×11008 的投影耗时 1.83 ms，其中 **1.71 ms 是解量化**。

融合 kernel 直接读打包的 nibble，在寄存器里算 `(q - zero) * scale`，直接喂 `tl.dot`。通过真实 `QuantLinear.forward` 测：一步里 7 个投影从 **200.2 ms → 12.0 ms**（36 层，**16.7×**）。整个 decode 从 **4.4 → 51.1 tok/s**。

**(b) `BK == group_size` 是那个"值 8×"的选择。** 让 `BK` 等于分组大小（128），scales 和 zeros 就能**作为向量一次加载并广播**，而不是逐元素 gather。这一条单独就值约 **8 倍**。

**(c) GEMV：`tl.dot` 在 M=1 下是纯浪费。** tiled kernel 因为 `tl.dot` 要求 `BM >= 16`，一个 decode 步会**算 16 行、扔掉 15 行**——在一个内存受限的步上做 16 倍无效 FLOPs。

GEMV 路径丢掉 `tl.dot`，改成对**一行激活**做 broadcast-multiply-reduce，并用代数改写把 scale/zero 从"每个权重一次"降成"每个输出列一次"：

```
sum_k x_k*(q-z)*s  ==  s*(sum_k x_k*q - z*sum_k x_k)
```

交错 A/B 实测（消掉笔记本时钟漂移）：一步的 252 个投影 **8.4 → 7.3 ms，186 → 202 GB/s**，每轮都成立。

**(d) tile 策略按带宽调。** M=1 时窄 `BN=32` 才能保住足够多 CTA 打满带宽（`BN=128` 会让 256 行的 k/v 投影只剩 2 个 CTA），这让单层 7 个投影从 383 us 降到 195 us。

### 4.5 为什么三个开关必须一起开（乘法，不是加法）

| 场景 | 只开 graph | 只开融合 attention | 三个一起 |
|---|---|---|---|
| 512 | 51.3 → 61.8 ✅ | 52.1 → 50.5 ❌ | 52.1 → **73.2** |
| 2048 | 47.9 → 45.4 ❌ | 47.9 → 49.7 ✅ | 47.9 → **67.8** |
| 4096 | — | — | 39.9 → **64.5** |

规律：**graph 只在 CPU 是瓶颈时有用，融合 attention 只在 GPU 是瓶颈时有用**。而它们各自都会把系统推向对方的领地——融合 attention 单独用会**增加 CPU 发射**（每层 2 次 launch 换掉 1 次 SDPA），所以在短 context 亏；graph 单独用会**暴露 GPU 工作量**，在长 context 亏。

**合起来才闭合**：graph 把融合 kernel 多出来的发射成本彻底吃掉，于是融合 kernel 的 GPU 收益可以纯赚。结果是 **+40% / +42% / +62%**，而且长 context 的衰减也被抹平——512→4096 的成本从 1.31× 降到 **1.14×**，因为随 context 增长的那部分（gather + padded attention）正是被消掉的东西。

### 4.6 贯穿所有决策的主线：decode 是 memory-bound

一步的 252 个 kernel 要读 **1.474 GB** 量化权重，8.4 ms 读完 = **186 GB/s**；而同一张卡上**未量化**的 `lm_head` gemv 能跑到 **372 GB/s**。所以判据不是"算得快不快"，而是"**字节搬得少不少**"。

这也解释了为什么几个实验被否掉：

- **组主布局**（`[K/G, N, G/2]`）：纯加载探针 256 → 277 GB/s，约 **+8%**。但要额外一份打包权重的拷贝（8 GB 卡上约 1.4 GB），且参考解量化器读不了这个布局。**不值得**。
- **转置 scales/zeros**：预期把 32 字节跨步的 2 字节 gather 变连续，实测 7.70 → 7.59 ms，**1.4%**——sector 浪费根本不是瓶颈。假设错了，**否掉**。
- **tile sweep**：在一步的 252 个 GEMM 的**聚合**上扫 `(BM, BN, num_stages, num_warps)`，现有策略赢（7.77 vs 8.04 ms）。

### 4.7 方法论：两条原则

**① 逐形状微基准会骗人。** 单个 12 MB 权重在重复调用中会留在 L2，于是测出 368 GB/s——那是 **L2 带宽，不是 DRAM**。必须**整步聚合**测。

**② 测量噪声必须用交错 A/B 压掉。** 同一个 kernel 单发测出过 7.16 / 7.59 / 7.70 ms（7% 抖动）。"8.4 → 7.3 ms"这个结论是靠 4 轮交错 A/B 得出的。

**③ 自检是加速的一部分，不是负担。** 每个 opt-in 开关启动时都拿参考路径对一遍，不合格就**拒绝并回退**：

- INT4 kernel：判据是**相对**误差 < 2e-2。用相对而非绝对，是因为参考本身在 bf16 里算，bf16 自身舍入噪声就有 ~1e-2；用绝对阈值会把一个差异小于 1 个 bf16 ulp 的 kernel 误判掉。
- CUDA graph：判据是 `verify_tolerance=0.0`，实测差异正好 **0.0**。
- 融合 attention：`verify_against_reference` 按 `text.py` 的方式重建参考路径（gather + length mask + SDPA），而不是调用会漏掉 padding 的 `attention.paged_attention`。

理由很实在：一个**悄悄算错**的 kernel 照样会输出流畅的句子。

---

## 五、性能数据

RTX 5060（8 GB）+ Qwen2.5-VL-3B INT4/group-128，batch 1，生成 128 token（`bench/run_bench.py`）：

| prompt | 加速前 decode | 加速后 decode | TTFT 前 | TTFT 后 |
|---|---|---|---|---|
| 512 | 52.1 tok/s | **73.2 tok/s** | 336 ms | 223 ms |
| 2,048 | 47.9 tok/s | **67.8 tok/s** | 832 ms | 722 ms |
| 4,096 | 39.9 tok/s | **64.5 tok/s** | 1,908 ms | 1,791 ms |

即相对参考 INT4 路径 **+40% / +42% / +62%**。

分项贡献：

| 优化 | 效果 |
|---|---|
| sync-free paged attention | decode 33.7 → 51.1 tok/s；TTFT 626 → 333 ms |
| 融合 INT4 dequant+GEMM | decode 4.4 → 51.1 tok/s；prefill 33.0 → 180.6 tok/s |
| GEMV 路径（M=1） | 一步 252 投影 8.4 → 7.3 ms（186 → 202 GB/s） |
| CUDA graph（单独） | 512: 51.3 → 61.8；2048: 47.9 → 45.4（变慢） |
| 融合 attention（单独） | 512: 52.1 → 50.5（变慢）；2048: 47.9 → 49.7 |
| 三者组合 | 见上表 |

摄像头实时描述：640×480 约 **510 ms/帧（1.9 fps）**；`--width 320` 或 `--max-vision-tokens 128` 约 **3.2 fps（+40%）**。

> 这些数字是**在 RTX 5060 上测的**。用 `bench/run_bench.py` 或 `qwen25vl-lite bench` 在你自己的硬件上复现。

---

## 六、验证了什么、没验证什么

这个仓库把"验证"当成设计的一部分，而不是贴在末尾的免责声明。

### 已数值验证（对照 HuggingFace `transformers`）

| 声明 | 检查方式 |
|---|---|
| MRoPE 与 HF **逐位一致** | `torch.equal`，零容差 |
| 视觉塔与 HF 一致 | `atol=1e-5` |
| 端到端 logits 与 HF 一致（纯文本与多模态） | `test_full_logits_match_hf` 等 |
| 分块 prefill == 单次 prefill | 20 种组合（chunk 16/8/5/3 × 长度 15/16/17/33/64） |
| paged decode == HF 增量解码（逐步） | `test_paged_decode_matches_hf_incremental` |
| 批量 decode == 逐条 decode（无 KV padding 泄漏） | `test_decode_batch_matches_independent_single_sequences` |
| 前缀缓存复用 == 重新计算 | `test_prefix_cache_reuses_and_matches_fresh_compute` |
| greedy 引擎输出 == `transformers.generate` | `test_greedy_generation_matches_hf_generate` |
| 融合 paged attention kernel == 参考路径 | 8 种形状（batch 1-4、head_dim 64/128/256、block 16/32、fp16/bf16） |
| CUDA graph decode == eager decode，**逐 token** | `tests/test_cuda_graph.py`，且捕获时差异恰好 0.0 |
| `/stats` 载荷可 JSON 序列化并报出实时速率 | `tests/test_engine.py` |
| 空闲时速率归零而非抱残值 | `test_decode_rate_falls_to_zero_once_the_engine_goes_idle` |
| `camera` 每帧发 JPEG data URL 并在服务器全拒时报错 | `tests/test_camera.py` |

### 明确没验证的

- **没有跨模型的质量评测**。量化、合并、剪枝、降分辨率都是**算法近似**，它们对具体任务的质量影响**没有被系统测过**。仓库里有 `scripts/quality_gate.py`（Gates A/B/C）就是给你自己测的。
- **FastV / `pool_merge` / `tome_merge` 的质量代价没有承诺**。`pool_merge` 和 `tome_merge` 实现了、有单测，但**没有接进任何档位**，README 也不声称它们可用——它们替换的是训练出来的特征几何，质量代价是模型相关的，必须逐模型实测。
- **`awq_style` 需要校准统计量**，没有就会静默退化成普通 groupwise INT4（代码里有注释说明，而不是假装）。
- **性能数字只在这台 RTX 5060 上**。
- **prefill 批处理没做**：prefill 一次只处理一个序列（异构长度的 prefill 不合并批处理）；decode **是**批处理的。continuous batching（同一次迭代里混合 prefill 和 decode、中途接纳新请求）已实现。

---

## 七、已知限制

- **prefill 一次一个序列**；decode 是批处理的。
- **FastV 要求整个 prompt 装进一个 chunk**，否则拒绝（而不是产出部分剪枝的序列）。
- **`float16` 在大图上会 NaN**（见 [1.4](#计算精度)），用 `bfloat16`。
- **无 tensor parallelism**。
- **CUDA graph 与自定义 kernel 都是 opt-in 且自检**；默认路径才是上面所有正确性声明描述的那条。
- 服务端做 chat 请求需要 tokenizer；`/v1/completions` 配 `prompt_token_ids` 不需要。
- 服务端是**单线程驱动引擎**的（`LLMEngine` 非线程安全）；请求线程只入队并等待条件变量，这正是它能批处理的原因。

---

## 八、测试

```bash
pip install -e ".[dev]"
python -m pytest tests/                        # 176 passed, 2 skipped（纯 CPU，无下载）
python -m pytest tests/test_model_vs_hf.py     # 只跑等价性门
python -m pytest tests/test_chunked_prefill.py
```

三个文件跑 GPU-only 路径，在纯 CPU 机器上跳过：`tests/test_int4_kernel.py`、`tests/test_paged_attn_kernel.py`、`tests/test_cuda_graph.py`。

测试套件需要 `transformers` 作为参考实现，并**在内存里现场构造一个随机权重的 tiny Qwen2.5-VL**——**不需要下载 checkpoint**。

所有等价性声明都是对着 **`transformers` 5.16.1** 测的，`dev` extra 正好钉住这个版本。参考实现允许在版本之间改动内部结构，5.20 就改了：重命名了 rotary embedding 类、把 `apply_multimodal_rotary_pos_emb` 换成通用 helper、把 MRoPE 的 interleave 挪进了 cos/sin 表。所以"与 transformers 一致"是**关于某个版本**的声明，不是移动靶。

---

## 许可证

Apache-2.0，见 `LICENSE`。
