#SPDX-License-Identifier: MIT
#Copyright (c) 2026 Jianxin Chen
"""TOLlm: a lightweight inference engine for large language models. The implementation will be iteratively layered by engine / core / attention / model."""

from tollm.llm import LLM
from tollm.sampling_params import SamplingParams

__version__ = "0.1.0"
__all__ = ["LLM", "SamplingParams", "__version__"]
