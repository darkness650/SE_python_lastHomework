from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseDetectorConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseDetector:
    """MediaPipe Pose 的薄封装。业务层只暴露 numpy 关键点，不暴露 MediaPipe 对象。"""

    def __init__(self, config: Optional[PoseDetectorConfig] = None):
        """初始化 PoseDetector。

        输入:
        - config: 可选的 PoseDetectorConfig，用于控制模型复杂度与置信度阈值。

        输出: 无（构造器）。

        作用: 延迟导入 mediapipe 并创建内部的 Pose 推理对象，用于后续帧的姿态检测。
        """
        self._config = config or PoseDetectorConfig()
        # 延迟导入，避免没有安装 mediapipe 时 import 直接炸
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self._config.model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )

    def detect_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """返回 (33,4) 的 numpy 数组：x,y,z,visibility，单位为归一化坐标。

        输入:
        - frame_bgr: BGR 格式的图像帧，numpy 数组，形状 (h,w,3)。

        输出:
        - 若检测到人体，返回 shape 为 (33,4) 的 numpy.ndarray（float32），每行为 x,y,z,visibility，均为归一化值；
        - 若未检测到人体或输入无效，返回 None。

        作用: 对单帧图像运行 MediaPipe Pose 推理并将结果转换为纯 numpy 格式，便于业务层使用。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if result.pose_landmarks is None:
            return None

        lm = result.pose_landmarks.landmark
        data = np.zeros((33, 4), dtype=np.float32)
        for i in range(33):
            data[i, 0] = lm[i].x
            data[i, 1] = lm[i].y
            data[i, 2] = lm[i].z
            data[i, 3] = lm[i].visibility
        return data

    def close(self) -> None:
        """释放内部 MediaPipe 资源。

        输入: 无
        输出: 无
        作用: 关闭并释放内部 Pose 对象所占资源，调用后不应再使用该实例进行推理。
        """
        self._pose.close()
