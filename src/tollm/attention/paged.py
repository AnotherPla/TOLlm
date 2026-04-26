"""PagedAttention 与 block table 的 Python/扩展入口（核心逻辑可后续下放到 Triton/C++）。"""

from __future__ import annotations

def attention_stub(*_args, **_kwargs) -> None:
    raise NotImplementedError("PagedAttention: 从 nano-vllm 或参考实现迁移/对齐")
