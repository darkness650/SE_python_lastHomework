from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# MediaPipe Pose landmark indices
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28


@dataclass(frozen=True)
class ScoringConfig:
    visibility_threshold: float = 0.5


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算 ∠ABC 的角度（度）。

    输入: a,b,c 为点坐标，形状 (2,) 或 (3,)。
    输出: 角度值（度），若不可计算返回 NaN。
    作用: 以向量夹角计算三点构成的关节角。
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-9:
        return float("nan")
    cosv = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _get_xy(lm: np.ndarray, idx: int) -> np.ndarray:
    """取关键点二维坐标。

    输入: lm 为 (33,4) 数组；idx 关键点索引。
    输出: (2,) numpy.ndarray，对应 (x,y)。
    作用: 从归一化关键点数据中提取二维位置用于角度计算。
    """
    return lm[idx, :2].astype(np.float32)


def _visible(lm: np.ndarray, idx: int, th: float) -> bool:
    """判断关键点可见性是否达阈值。

    输入: lm (33,4)、idx 索引、th 置信度阈值。
    输出: bool，表示该点是否有效。
    作用: 过滤低可见度关键点，避免噪声影响计算。
    """
    return float(lm[idx, 3]) >= th


def compute_joint_angles(lm: np.ndarray, visibility_threshold: float = 0.5) -> dict[str, float]:
    """从 (33,4) 关键点计算主要关节角度。

    输入: lm 为 (33,4) 的 numpy.ndarray；visibility_threshold 为可见性阈值。
    输出: 字典 {关节名: 角度(度)}；不可用则为 NaN。
    作用: 依据关键点位置计算肘、肩、髋、膝等关节角度。
    """
    th = visibility_threshold

    def angle_if_ok(i1: int, i2: int, i3: int) -> float:
        if not (_visible(lm, i1, th) and _visible(lm, i2, th) and _visible(lm, i3, th)):
            return float("nan")
        return _angle_deg(_get_xy(lm, i1), _get_xy(lm, i2), _get_xy(lm, i3))

    return {
        "l_elbow": angle_if_ok(L_SHOULDER, L_ELBOW, L_WRIST),
        "r_elbow": angle_if_ok(R_SHOULDER, R_ELBOW, R_WRIST),
        "l_shoulder": angle_if_ok(L_ELBOW, L_SHOULDER, L_HIP),
        "r_shoulder": angle_if_ok(R_ELBOW, R_SHOULDER, R_HIP),
        "l_knee": angle_if_ok(L_HIP, L_KNEE, L_ANKLE),
        "r_knee": angle_if_ok(R_HIP, R_KNEE, R_ANKLE),
        "l_hip": angle_if_ok(L_SHOULDER, L_HIP, L_KNEE),
        "r_hip": angle_if_ok(R_SHOULDER, R_HIP, R_KNEE),
    }


def compare_angles(
    user_lm: Optional[np.ndarray],
    ref_lm: Optional[np.ndarray],
    config: Optional[ScoringConfig] = None,
) -> tuple[float, dict[str, float]]:
    """对比用户与参考关键关节角度，输出 0~100 分。

    输入: user_lm/ref_lm 为 (33,4) 数组或 None；config 可选评分配置。
    输出: (score, details)：score 为 0~100 的相似度分；details 为每个关节的角度差（度）。
    作用: 计算两组姿态的主要关节角度差并线性映射到分数。
    """
    cfg = config or ScoringConfig()
    if user_lm is None or ref_lm is None:
        return 0.0, {}

    user = compute_joint_angles(user_lm, cfg.visibility_threshold)
    ref = compute_joint_angles(ref_lm, cfg.visibility_threshold)

    diffs: dict[str, float] = {}
    valid = 0
    total_penalty = 0.0

    for k, ua in user.items():
        ra = ref.get(k, float("nan"))
        if np.isnan(ua) or np.isnan(ra):
            continue
        d = float(abs(ua - ra))
        # 角度在 0~180，差值 > 180 不可能，但这里做个保护
        d = min(d, 180.0)
        diffs[k] = d
        valid += 1
        total_penalty += d

    if valid == 0:
        return 0.0, diffs

    avg_diff = total_penalty / valid
    # 简单线性映射：平均差 0° => 100分，60° => 0分（可后续调整）
    score = max(0.0, 100.0 * (1.0 - avg_diff / 60.0))
    return float(score), diffs
