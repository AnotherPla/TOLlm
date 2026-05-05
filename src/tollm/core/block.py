"""PagedAttention 的块元数据。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalBlock:
    block_id: int
    # ref_count, hash for prefix，后续按需加

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []