"""model config"""

import os
from transformers import AutoConfig
from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """only care about the shape and hyperparameters after parsing, the weight path is combined by the Engine"""
    model_path: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config:AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    
    def __post_init__(self):
        assert os.path.exists(self.model_path), f"Model path {self.model_path} does not exist"
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size must be divisible by 256"
        assert 1<=self.tensor_parallel_size<=8, "tensor_parallel_size must be between 1 and 8"
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
