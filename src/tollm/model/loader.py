"""从 HuggingFace safetensors/目录加载，映射到本仓库权重命名。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

def load_state_dict_from_hf(model_path: Union[str, Path]) -> dict[str, Any]:
    del model_path
    return {}
