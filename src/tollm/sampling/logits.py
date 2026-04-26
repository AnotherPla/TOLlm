"""对 logits 的后处理与采样，便于与引擎最后一步解耦、单测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SamplerState:
    seed: Optional[int] = None


def sample_greedy(logits_row: Any) -> int:
    del logits_row
    return 0
