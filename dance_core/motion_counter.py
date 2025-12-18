from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import cv2

import numpy as np

from .pose_detector import PoseDetector, PoseDetectorConfig
from .scoring import compute_joint_angles
from .types import PoseLandmarks


# 统一的角度键顺序（与 scoring.compute_joint_angles 对齐）
ANGLE_KEYS_ORDER: List[str] = [
    "l_elbow", "r_elbow", "l_shoulder", "r_shoulder",
    "l_knee", "r_knee", "l_hip", "r_hip",
]

@dataclass
class AngleTemplate:
    keys: List[str]
    angles: np.ndarray  # (T, K)
    valid: np.ndarray   # (T, K) bool

    @staticmethod
    def from_video(ref_path: str, sample_fps: float = 10.0, visibility_th: float = 0.5, keys: Optional[List[str]] = None) -> "AngleTemplate":
        keys_used = keys or ANGLE_KEYS_ORDER
        cap = cv2.VideoCapture(ref_path)
        if not cap or not cap.isOpened():
            raise RuntimeError("无法打开参考视频")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / max(1e-6, sample_fps))))
        pd = PoseDetector(PoseDetectorConfig())
        rows: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    lm = pd.detect_landmarks(frame)
                    if lm is None:
                        rows.append(np.full((len(keys_used),), np.nan, dtype=np.float32))
                        valids.append(np.zeros((len(keys_used),), dtype=bool))
                    else:
                        angs = compute_joint_angles(lm, visibility_threshold=visibility_th)
                        arr = np.array([float(angs.get(k, np.nan)) for k in keys_used], dtype=np.float32)
                        rows.append(arr)
                        valids.append(~np.isnan(arr))
                idx += 1
        finally:
            try:
                cap.release()
                pd.close()
            except Exception:
                pass
        if not rows:
            raise RuntimeError("参考视频未抽取到角度模板")
        angles = np.stack(rows, axis=0)
        valid = np.stack(valids, axis=0)
        return AngleTemplate(keys=keys_used.copy(), angles=angles, valid=valid)


@dataclass
class TemplateMatcherConfig:
    tolerance_deg: float = 10.0
    visibility_th: float = 0.5
    lookahead: int = 3  # 基础前瞻步数（当未提供关键动作数时）


class TemplateRepetitionCounter:
    """基于参考模板的逐帧匹配计数器（支持动态前瞻对齐）：
    - 每帧与模板当前帧比较，不通过则在允许范围内向后前瞻；
    - 若提供关键动作数，则前瞻上限=到“下一个关键动作”的中心位置；
    - 匹配成功推进到匹配帧的下一帧；模板完整走一轮即计数+1。
    """

    def __init__(self, template: AngleTemplate, config: Optional[TemplateMatcherConfig] = None, key_actions: Optional[int] = None) -> None:
        self.template = template
        self.cfg = config or TemplateMatcherConfig()
        self.T, self.K = int(template.angles.shape[0]), int(template.angles.shape[1])
        self.idx = 0
        self.count = 0
        self.key_actions = key_actions if (key_actions is not None and key_actions > 0) else None

    def reset(self) -> None:
        self.idx = 0
        self.count = 0

    def _dynamic_lookahead(self) -> int:
        """计算当前帧允许的最大前瞻步数。
        若未配置 key_actions，则使用 cfg.lookahead；
        否则：以段长 seg_len=T/key_actions，将上限设置为到“下一个关键动作中心”的环形距离（允许跨越模板末尾）。
        """
        if not self.key_actions:
            return int(max(0, self.cfg.lookahead))
        if self.key_actions <= 0 or self.T <= 0:
            return int(max(0, self.cfg.lookahead))
        seg_len = float(self.T) / float(self.key_actions)
        seg_idx = int(self.idx // seg_len)
        center_next_nominal = (seg_idx + 1.5) * seg_len  # 下一个关键动作中心（可能超出 T-1）
        center_idx = int(round(center_next_nominal))
        # 环形距离：允许跨越末尾进入下一轮
        dist = (center_idx - self.idx) % self.T
        return int(max(0, dist))

    def update(self, lm: Optional[PoseLandmarks]) -> tuple[int, dict]:
        info = {"idx": self.idx, "T": self.T, "passed": False, "diffs": None, "skipped": 0}
        if lm is None:
            return self.count, info
        # 当前帧角度
        angs = compute_joint_angles(lm.data, visibility_threshold=self.cfg.visibility_th)
        user_vec = np.array([float(angs.get(k, np.nan)) for k in self.template.keys], dtype=np.float32)

        # 尝试匹配 当前模板帧 -> 向后若干帧（支持环形前瞻）
        def match_at(t_idx: int) -> tuple[bool, Optional[np.ndarray]]:
            ref_idx = t_idx % self.T
            ref_vec = self.template.angles[ref_idx]
            ref_valid = self.template.valid[ref_idx]
            both_valid = (~np.isnan(user_vec)) & ref_valid
            if not np.any(both_valid):
                return False, None
            diffs = np.abs(user_vec[both_valid] - ref_vec[both_valid])
            ok = bool(np.all(diffs <= self.cfg.tolerance_deg))
            return ok, diffs

        matched = False
        adv = 0
        ok0, diffs0 = match_at(self.idx)
        if ok0:
            matched = True
            adv = 1
            info["diffs"] = diffs0
        else:
            max_look = int(max(0, self._dynamic_lookahead()))
            for j in range(1, max_look + 1):
                ok_j, diffs_j = match_at(self.idx + j)
                if ok_j:
                    matched = True
                    adv = j + 1  # 前进到匹配帧的下一帧
                    info["diffs"] = diffs_j
                    info["skipped"] = j
                    break
        if matched:
            new_linear = self.idx + adv
            if self.T > 0:
                wraps = new_linear // self.T
                self.count += wraps
                self.idx = new_linear % self.T
            else:
                self.idx = 0
            info["passed"] = True
        info["idx"] = self.idx
        return self.count, info
