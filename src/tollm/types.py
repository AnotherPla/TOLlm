"""跨模块共用的轻量类型别名与协议占位（后续可换成 Protocol）。"""

from __future__ import annotations

from typing import TypeAlias

Tensor: TypeAlias = "object"  # 使用 torch 时改为 torch.Tensor 或 npt.NDArray 等
