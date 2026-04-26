"""请求调度与连续批处理（stub，后续可换 v1 策略、chunked prefill）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from tollm.core.sequence import Sequence


@dataclass
class ScheduleBatch:
    """当前步要喂给模型的 batch。"""

    seqs: List[Sequence] = field(default_factory=list)
    is_prefill: bool = True


class Scheduler:
    def __init__(self, max_num_seqs: int = 256) -> None:
        self._max = max_num_seqs
        self._waiting: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self._waiting.append(seq)

    def schedule(self) -> ScheduleBatch:
        if not self._waiting:
            return ScheduleBatch()
        return ScheduleBatch(seqs=self._waiting[: self._max], is_prefill=True)
