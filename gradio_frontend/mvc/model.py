from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from dance_core.types import PoseLandmarks
from dance_core.pose_connections import POSE_CONNECTIONS

# 将临时目录设置为项目根目录下的 .temp
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEMP_DIR = os.path.join(PROJECT_ROOT, ".temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def save_uploaded(video: Optional[str], name: str) -> Optional[str]:
    """将上传的文件复制到 项目根目录/.temp 下并返回新路径。"""
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


def _compute_angle_deg(landmarks: PoseLandmarks,
                       a: int, b: int, c: int) -> Optional[float]:
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
