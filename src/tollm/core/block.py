"""PagedAttention 的块元数据。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalBlock:
    block_id: int
    # ref_count, hash for prefix，后续按需加
