"""用户可见的 LLM 类；内部委托 Runner / Engine。"""
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from tollm.config import ModelConfig
from tollm.sampling import SamplingParams
from tollm.core.scheduler import Scheduler
from tollm.core.sequence import Sequence
from tollm.engine.runner import Runner

class LLM_Engine:
    """推理入口桩：接好 `model_path` 后接 loader 与执行循环。"""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        config_fields = {field.name for field in fields(ModelConfig)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = ModelConfig(model_path, **config_kwargs)
        self._model_path = model_path
        self._kwargs = kwargs

    def generate(self, prompts: list[str], params: SamplingParams) -> list[dict[str, Any]]:
        del params
        return [{"text": f"<stub> for {p!r}"} for p in prompts]
