"""块的分配/回收，可逐步加前缀缓存、复制写等。"""

from __future__ import annotations

from typing import List, Optional

from tollm.core.block import PhysicalBlock


class BlockManager:
    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self._free: List[int] = list(range(num_blocks))

    def allocate(self) -> Optional[PhysicalBlock]:
        if not self._free:
            return None
        bid = self._free.pop()
        return PhysicalBlock(block_id=bid)

    def free(self, block: PhysicalBlock) -> None:
        self._free.append(block.block_id)
