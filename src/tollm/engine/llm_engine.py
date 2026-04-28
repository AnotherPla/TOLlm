"""用户可见的 LLM 类；内部委托 Runner / Engine。"""

from __future__ import annotations

from typing import Any

from tollm.sampling import SamplingParams


class LLM_Engine:
    """推理入口桩：接好 `model_path` 后接 loader 与执行循环。"""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self._model_path = model_path
        self._kwargs = kwargs

    def generate(self, prompts: list[str], params: SamplingParams) -> list[dict[str, Any]]:
        del params
        return [{"text": f"<stub> for {p!r}"} for p in prompts]
