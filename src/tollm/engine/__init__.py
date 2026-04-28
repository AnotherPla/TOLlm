"""对外入口：LLM 与 Engine 的薄封装，接口未来可对齐 vLLM / 上游 nano API。"""

from tollm.engine.llm_engine import LLM_Engine
from tollm.sampling import SamplingParams

__all__ = ["LLM_Engine", "SamplingParams"]
