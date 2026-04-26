"""单步/多步执行循环、与调度器、KV 的粘合（从 llm 中拆出便于阅读与测试）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
