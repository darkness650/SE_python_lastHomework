from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from dance_core.pose_detector import PoseDetector, PoseDetectorConfig
from dance_core.types import PoseLandmarks
from dance_core.pose_connections import POSE_CONNECTIONS

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 名称到索引（MediaPipe Pose 33点）
JOINT_NAME_TO_INDEX = {
    "鼻尖": 0,
    "左眼内角": 1, "左眼": 2, "左眼外角": 3, "右眼内角": 4, "右眼": 5, "右眼外角": 6,
    "左耳": 7, "右耳": 8,
    "口左": 9, "口右": 10,
    "左肩": 11, "右肩": 12,
    "左肘": 13, "右肘": 14,
    "左腕": 15, "右腕": 16,
    "左小指": 17, "右小指": 18,
    "左食指": 19, "右食指": 20,
    "左拇指": 21, "右拇指": 22,
    "左髋": 23, "右髋": 24,
    "左膝": 25, "右膝": 26,
    "左踝": 27, "右踝": 28,
    "左脚跟": 29, "右脚跟": 30,
    "左脚尖": 31, "右脚尖": 32,
}

# 参考：候选三元组（覆盖手臂与腿部的常用关节）
CANDIDATE_TRIPLETS: list[Tuple[int, int, int]] = [
    # 右臂、左臂
    (12, 14, 16), (11, 13, 15),
    # 右腿、左腿
    (24, 26, 28), (23, 25, 27),
]


def save_uploaded(video: Optional[str], name: str) -> Optional[str]:
    """将上传的文件复制到 .temp 下并返回新路径。"""
    if not video:
        return None
    basename = os.path.basename(video)
    target = os.path.join(TEMP_DIR, f"{name}_{basename}")
    try:
        if os.path.abspath(video) != os.path.abspath(target):
            with open(video, "rb") as fsrc, open(target, "wb") as fdst:
                fdst.write(fsrc.read())
    except Exception:
        return None
    return target


def _compute_angle_deg(landmarks: PoseLandmarks, a: int, b: int, c: int) -> Optional[float]:
    """计算三点夹角（度）。返回None表示关键点不可见或无效。"""
    data = landmarks.data
    vis_ok = (data[a, 3] >= 0) and (data[b, 3] >= 0) and (data[c, 3] >= 0)
    if not vis_ok:
        return None
    pa = data[a, :2]
    pb = data[b, :2]
    pc = data[c, :2]
    v1 = pa - pb
    v2 = pc - pb
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosang)))
    return angle


def _collect_angles_from_video(video_path: str,
                               triplet: Tuple[int, int, int],
                               min_visible: float,
                               sample_fps: float = 10.0) -> list[float]:
    """以固定采样率遍历视频并收集该三元组的角度。"""
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(1e-6, sample_fps))))
    pd = PoseDetector(PoseDetectorConfig())
    angles: list[float] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                lm = pd.detect_landmarks(frame)
                if lm is not None:
                    plm = PoseLandmarks(lm)
                    a, b, c = triplet
                    if plm.data[a, 3] >= min_visible and plm.data[b, 3] >= min_visible and plm.data[c, 3] >= min_visible:
                        ang = _compute_angle_deg(plm, a, b, c)
                        if ang is not None:
                            angles.append(ang)
            idx += 1
    finally:
        try:
            cap.release()
            pd.close()
        except Exception:
            pass
    return angles


def select_best_triplet_from_reference(ref_path: str, min_visible: float) -> Optional[Tuple[int, int, int]]:
    """从参考视频在候选集合中选择角度方差最大的三元组。"""
    best_triplet = None
    best_var = -1.0
    for tri in CANDIDATE_TRIPLETS:
        angles = _collect_angles_from_video(ref_path, tri, min_visible)
        if len(angles) >= 5:
            var = float(np.var(np.array(angles)))
            if var > best_var:
                best_var = var
                best_triplet = tri
    return best_triplet


def calibrate_thresholds_from_reference(ref_path: Optional[str],
                                        triplet: Tuple[int, int, int],
                                        min_visible: float) -> Optional[Tuple[float, float]]:
    """遍历参考视频，统计角度范围并给出低/高阈值建议。
    策略：在可见度>=min_visible的帧上统计角度的5/95分位作为低/高阈值。
    返回(None)表示无法校准。"""
    if not ref_path:
        return None
    cap = cv2.VideoCapture(ref_path)
    if not cap or not cap.isOpened():
        return None
    angles: list[float] = []
    pd = PoseDetector(PoseDetectorConfig())
    a, b, c = triplet
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            lm = pd.detect_landmarks(frame)
            if lm is None:
                continue
            plm = PoseLandmarks(lm)
            # 可见度过滤
            if plm.data[a, 3] < min_visible or plm.data[b, 3] < min_visible or plm.data[c, 3] < min_visible:
                continue
            angle = _compute_angle_deg(plm, a, b, c)
            if angle is not None:
                angles.append(angle)
        if len(angles) < 5:
            return None
        arr = np.array(angles)
        low = float(np.percentile(arr, 5))
        high = float(np.percentile(arr, 95))
        # 保证间隔合理
        if high - low < 10:
            # 扩展一个固定边界
            mid = (high + low) / 2
            low = max(0.0, mid - 20)
            high = min(180.0, mid + 20)
        return low, high
    finally:
        try:
            cap.release()
            pd.close()
        except Exception:
            pass


def _draw_annotations(frame: np.ndarray,
                      landmarks: Optional[PoseLandmarks],
                      triplet: Tuple[int, int, int],
                      angle: Optional[float]) -> np.ndarray:
    """在帧上绘制关键点、全身骨架与角度文字（包含多关节角度）。"""
    vis = frame.copy()
    color_pts = (0, 255, 0)
    color_line = (255, 200, 0)
    color_text = (0, 0, 255)
    color_skeleton = (150, 150, 150)
    thickness = 2
    h, w = vis.shape[:2]

    def to_xy(idx: int):
        return (int(landmarks.data[idx, 0] * w), int(landmarks.data[idx, 1] * h))

    if landmarks is not None:
        # 画全身骨架
        for a, b in POSE_CONNECTIONS:
            try:
                if landmarks.data[a, 3] >= 0 and landmarks.data[b, 3] >= 0:
                    cv2.line(vis, to_xy(a), to_xy(b), color_skeleton, 2)
            except Exception:
                pass
        # 高亮当前三元组关键点与连线
        a, b, c = triplet
        for idx in [a, b, c]:
            if landmarks.data[idx, 3] >= 0:
                x, y = to_xy(idx)
                cv2.circle(vis, (x, y), 6, color_pts, -1)
        try:
            cv2.line(vis, to_xy(a), to_xy(b), color_line, thickness)
            cv2.line(vis, to_xy(b), to_xy(c), color_line, thickness)
        except Exception:
            pass

    # 角度文字：主角度 + 多关节角度（上肢与下肢）
    text_lines = []
    text_lines.append(f"主角度: {angle:.1f}°" if angle is not None else "主角度: N/A")
    if landmarks is not None:
        # 定义多关节三元组：双臂与双腿
        multi_triplets = [
            (12, 14, 16),  # 右臂
            (11, 13, 15),  # 左臂
            (24, 26, 28),  # 右腿
            (23, 25, 27),  # 左腿
        ]
        names = ["右臂", "左臂", "右腿", "左腿"]
        for name, tri in zip(names, multi_triplets):
            ang = _compute_angle_deg(landmarks, *tri)
            text_lines.append(f"{name}:{ang:.1f}°" if ang is not None else f"{name}:N/A")
    # 绘制文字块
    y0 = 30
    for i, t in enumerate(text_lines):
        cv2.putText(vis, t, (20, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2, cv2.LINE_AA)
    return vis

