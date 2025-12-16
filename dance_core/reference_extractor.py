from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .pose_detector import PoseDetector
from .types import PoseFrame, PoseLandmarks


@dataclass(frozen=True)
class ExtractConfig:
    sample_fps: float = 10.0
    max_seconds: Optional[float] = None


class ReferenceExtractionError(RuntimeError):
    pass


def extract_reference_sequence(
    video_path: str,
    detector: PoseDetector,
    config: Optional[ExtractConfig] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[PoseFrame]:
    """从参考视频中抽取姿态序列。

    输入:
    - video_path: 参考视频文件路径。
    - detector: PoseDetector 实例，用于逐帧检测关键点。
    - config: 可选的抽样配置（帧率、最长时长）。
    - on_progress: 可选进度回调，形如 (cur_frames, total_frames)。

    输出:
    - 返回按时间顺序的 PoseFrame 列表，每个包含时间戳与可选的关键点（33x4）。

    作用:
    - 以指定采样帧率遍历视频，将检测到的姿态关键点打包为 PoseFrame 并按时间顺序返回；
    - 未能打开视频或未抽取到任何帧时抛出 ReferenceExtractionError。
    """
    cfg = config or ExtractConfig()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ReferenceExtractionError(f"无法打开视频：{video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-6:
        src_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / max(1e-6, cfg.sample_fps))))

    frames: list[PoseFrame] = []
    idx = 0

    max_frames = None
    if cfg.max_seconds is not None:
        max_frames = int(cfg.max_seconds * src_fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames is not None and idx >= max_frames:
                break

            if idx % step == 0:
                ts = idx / src_fps
                lm = detector.detect_landmarks(frame)
                pf = PoseFrame(
                    timestamp_s=float(ts),
                    landmarks=PoseLandmarks(lm) if lm is not None else None,
                )
                frames.append(pf)

            idx += 1

            if on_progress is not None and total_frames > 0 and idx % 10 == 0:
                on_progress(min(idx, total_frames), total_frames)

    finally:
        cap.release()

    if len(frames) == 0:
        raise ReferenceExtractionError("参考视频未抽取到任何帧")

    return frames


def find_reference_by_time(sequence: list[PoseFrame], t_s: float) -> Optional[np.ndarray]:
    """根据时间戳获取参考姿态关键点 (33,4)。

    输入:
    - sequence: 由 extract_reference_sequence 生成的 PoseFrame 列表（时间有序）。
    - t_s: 查询的时间戳（秒）。

    输出:
    - 若附近存在带 landmarks 的帧，返回 (33,4) numpy.ndarray；否则返回 None。

    作用:
    - 在时间轴上查找相邻两帧并进行线性插值，减少低采样率带来的跳变；
    - 若任一侧不可用，则回退到距离最近且可用的单帧。
    """
    if not sequence:
        return None

    # 假设 sequence 按时间顺序（extract_reference_sequence 本身是顺序追加）
    times = [pf.timestamp_s for pf in sequence]
    i = bisect.bisect_left(times, t_s)

    if i <= 0:
        first = sequence[0]
        return None if first.landmarks is None else first.landmarks.data
    if i >= len(sequence):
        last = sequence[-1]
        return None if last.landmarks is None else last.landmarks.data

    prev = sequence[i - 1]
    nxt = sequence[i]

    # 若两侧都有 landmarks，做线性插值
    if prev.landmarks is not None and nxt.landmarks is not None:
        t0 = prev.timestamp_s
        t1 = nxt.timestamp_s
        denom = t1 - t0
        if denom <= 1e-9:
            return prev.landmarks.data
        alpha = float((t_s - t0) / denom)
        alpha = max(0.0, min(1.0, alpha))
        return (1.0 - alpha) * prev.landmarks.data + alpha * nxt.landmarks.data

    # 否则退化为“更近的那一帧（且有 landmarks）”
    cand = []
    if prev.landmarks is not None:
        cand.append((abs(prev.timestamp_s - t_s), prev.landmarks.data))
    if nxt.landmarks is not None:
        cand.append((abs(nxt.timestamp_s - t_s), nxt.landmarks.data))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[0][1]
