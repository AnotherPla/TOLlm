"""FlashAttention / SDPA 等封装，与设备能力探测。"""

from __future__ import annotations

def flash_attention_stub(*_args, **_kwargs) -> None:
    raise NotImplementedError("FlashAttention: 接 torch.sdpa 或官方扩展")
