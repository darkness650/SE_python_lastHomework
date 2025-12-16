from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PoseLandmarks:
    """33个关键点，形状为 (33, 4)：x,y,z,visibility（均为float）。

    属性:
    - data: numpy.ndarray，形状 (33,4)，每行为 x,y,z,visibility（归一化坐标）。
    """

    data: np.ndarray  # (33, 4)

    @property
    def xy(self) -> np.ndarray:
        """返回二维坐标部分。

        输入: 无（通过实例访问）。
        输出: numpy.ndarray，形状 (33,2)，对应每个关键点的 (x,y)。
        作用: 方便只使用二维坐标的下游计算（如绘图/角度计算）。
        """
        return self.data[:, :2]


@dataclass(frozen=True)
class PoseFrame:
    timestamp_s: float
    landmarks: Optional[PoseLandmarks]


@dataclass(frozen=True)
class SimilarityResult:
    score_0_100: float
    details: dict[str, float]
