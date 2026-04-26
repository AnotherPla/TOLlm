"""LLaMA 风格 Transformer（与 nano 对齐时可逐步从 stub 补全）。"""

from __future__ import annotations

from typing import Any

from tollm.model.base import BaseModel


class LlamaForCausalLM(BaseModel):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError
