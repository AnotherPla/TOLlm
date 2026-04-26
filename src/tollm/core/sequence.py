"""单条请求的状态机（prompt + 已生成长度 + 状态）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    DECODE = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    generated: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)
