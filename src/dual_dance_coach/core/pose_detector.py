from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from dual_dance_coach.view.resources import resolve_blob_path

# mediapipe 模型文件路径
MODEL_PATH = resolve_blob_path("pose_landmarker_heavy.task")


@dataclass(frozen=True)
class PoseDetectorConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseDetector:
    """MediaPipe Pose 的薄封装。业务层只暴露 numpy 关键点，不暴露 MediaPipe 对象。"""

    def __init__(self, config: PoseDetectorConfig | None = None):
        """初始化 PoseDetector。

        输入:
        - config: 可选的 PoseDetectorConfig，用于控制模型复杂度与置信度阈值。

        输出: 无（构造器）。

        作用: 导入 mediapipe 并创建内部的 Pose 推理对象，用于后续帧的姿态检测。
        """
        self._config = config or PoseDetectorConfig()

        if not MODEL_PATH.exists():
            raise RuntimeError(
                "未找到模型文件："
                f"{MODEL_PATH}。请将 pose_landmarker_heavy.task 放在可执行程序或项目根目录下。"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self._config.min_detection_confidence,
            min_pose_presence_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)

    def detect_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._pose.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        data = np.zeros((len(lm), 4), dtype=np.float32)
        for i, point in enumerate(lm):
            data[i, 0] = point.x
            data[i, 1] = point.y
            data[i, 2] = point.z
            data[i, 3] = getattr(point, "visibility", 0.0)
        return data

    def close(self) -> None:
        """释放内部 MediaPipe 资源。

        输入: 无
        输出: 无
        作用: 关闭并释放内部 Pose 对象所占资源，调用后不应再使用该实例进行推理。
        """
        self._pose.close()
