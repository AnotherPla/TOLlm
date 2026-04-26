"""KV Cache 张量布局与 in-place 更新接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class KVLayout:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    # max_blocks 等


class KVCache:
    """存根：后续可换 Paged 或 vLLM 式 tensor。"""

    def __init__(self, layout: KVLayout) -> None:
        self._layout = layout

    def as_tensors(self) -> Optional[List[Any]]:
        return None
