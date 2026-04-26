"""模型与运行时配置（数据类/字典），与 nano-vllm 的 ModelConfig 思路一致，按需扩展。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """推理侧只关心「已解析」后的形状与超参，权重路径由 Engine 组合输入。"""

    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 4096
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    torch_dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict)
