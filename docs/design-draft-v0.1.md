# qwen25vl-lite — 设计草案 v0.1（外部评审用）

**目标**：在 **RTX 5060 8GB VRAM + 96GB 系统内存** 上运行 Qwen2.5-VL-3B / 7B，最大化
可服务的上下文长度与吞吐，并提供 OpenAI 兼容 API。硬件为 Blackwell sm_120，8GB 显存是硬瓶颈。

## 一、显存预算（先算账，再定架构）

| 项 | 3B | 7B |
|---|---|---|
| BF16 权重 | 6.2 GB | 15.4 GB |
| INT8 weight-only（per-channel） | 3.4 GB | 8.2 GB |
| INT4 groupwise(g=128) | 2.1 GB | 4.4 GB |
| vision tower（0.44B） | 0.9 GB（fp16） | 0.9 GB |
| KV / token（fp16，36L/2KVh/128d） | 18 KB | 18 KB |

KV 池若给 3.5 GB：fp16 约 125k token；KV INT8 约 250k token。
**结论**：3B 必须 INT8/INT4 才装得下；7B 必须 INT4 + 部分 CPU offload。

## 二、架构（模块清单）

```
config.py       HF config 解析（含 vision_config / mrope_section）
rope.py         MRoPE(3 段 16/24/24) + vision 2D RoPE
quant.py        INT8 per-channel / INT4 groupwise weight-only + KV INT8
vision.py       ViT(depth32, window112, fullatt@7/15/23/31) + PatchMerger(2x2)
text.py         Qwen2 解码层（GQA + RMSNorm + SwiGLU）
model.py        加载 HF safetensors，权重名对齐，forward
cache.py        Paged KV（块表+引用计数）+ radix 前缀缓存
vision_budget.py 动态分辨率上限 + ToMe 合并 + FastV 剪枝（三档 conservative/balanced/aggressive）
planner.py      显存预算规划器：读 free VRAM → 定 KV 块数 / vision 落 CPU 与否 / prefill chunk
engine.py       连续批处理 + chunked prefill + 抢占 swap
server.py       OpenAI 兼容 /v1/chat/completions (SSE) + /health /stats
cli.py          命令行推理
```

## 三、核心决策（需要外部评审）

1. **权重量化放在离线（safetensors 直读 + 量化缓存）**，而不是运行时 —— 避免每次启动重算。
2. **INT4 反量化在 PyTorch 层做（dequant → F.linear），不写融合 kernel**。
   收益定位为「显存」而非「算力」；融合 kernel 留 v0.2。风险：算力可能反而变慢。
3. **视觉 token 预算**：默认 `balanced` = 分辨率上限 1M px + ToMe(r=2) 合并；
   `aggressive` 再加 FastV 剪枝（保留 50%）。风险：精度损失，需回归测试兜底。
4. **KV 用 paged + 前缀缓存**，图像前缀跨请求复用（多轮 VL 对话主场景）。
5. **attention 走 `F.scaled_dot_product_attention`**（sm_120 无 flash-attn 轮子时自动退 mem-efficient）。
6. **CPU 回退**：全部模块 device-agnostic，CI 跑 tiny config 正确性测试（对比 HF transformers）。

## 四、自担风险点（请重点评审）

- R1：INT4 逐层反量化的实际吞吐收益可能为负（带宽省了、算力多了）——是否该默认 INT8？
- R2：视觉 token 剪枝对 OCR/小字类任务的精度损伤可能不可接受 —— 默认档位怎么定？
- R3：8GB 上 7B + 长上下文是否现实，还是应该只支持 3B？
- R4：无 GPU 时 CI 只能验证 tiny config 的数值一致性，无法验证真实加速比 —— 如何让验收更有说服力？
- R5：paged KV + 前缀缓存在单卡 8GB 上的收益是否被块碎片抵消？
