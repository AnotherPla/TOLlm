"""用户可见的 LLM 类；内部委托 Runner / Engine。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    top_k: int = -1
    extra: dict[str, Any] = field(default_factory=dict)


class LLM:
    """推理入口桩：接好 `model_path` 后接 loader 与执行循环。"""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self._model_path = model_path
        self._kwargs = kwargs

    def generate(self, prompts: list[str], params: SamplingParams) -> list[dict[str, Any]]:
        del params
        return [{"text": f"<stub> for {p!r}"} for p in prompts]
