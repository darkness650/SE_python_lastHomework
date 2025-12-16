from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2

import numpy as np

from .pose_detector import PoseDetector, PoseDetectorConfig
from .scoring import compute_joint_angles
from .types import PoseLandmarks


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算夹角 ∠ABC 的角度（度）。
    a,b,c 为 (x,y) 点。
    """
    ba = a - b
    bc = c - b
    # 防止零向量
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return float("nan")
    cosang = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


@dataclass
class CounterConfig:
    # 使用的关节三点（默认右肘：肩(12)-肘(14)-腕(16)），可换为膝等
    joint_a: int = 12
    joint_b: int = 14
    joint_c: int = 16
    # 阈值：下（收缩）与上（伸展）角度
    angle_low_deg: float = 50.0
    angle_high_deg: float = 160.0
    # 去抖动与有效性
    min_visible: float = 0.5  # 三点最小可见度
    min_frames_between: int = 3  # 相邻状态变更的最少帧数


class RepetitionCounter:
    """简单的重复计数器：基于一个关节角度在低/高阈值间摆动，统计完整次数。

    定义：一次完整重复 = 从高(伸展) -> 低(收缩) -> 高(伸展)。
    """

    def __init__(self, config: Optional[CounterConfig] = None) -> None:
        self.config = config or CounterConfig()
        self._state = "unknown"  # "high" | "low" | "unknown"
        self._last_switch_frame_idx = -999
        self.count = 0
        self.frame_idx = 0

    def reset(self) -> None:
        self._state = "unknown"
        self._last_switch_frame_idx = -999
        self.count = 0
        self.frame_idx = 0

    def _visible_ok(self, lm: PoseLandmarks) -> bool:
        vis = lm.data[[self.config.joint_a, self.config.joint_b, self.config.joint_c], 3]
        return bool(np.all(vis >= self.config.min_visible))

    def _current_angle(self, lm: PoseLandmarks) -> float:
        xy = lm.xy
        a = xy[self.config.joint_a]
        b = xy[self.config.joint_b]
        c = xy[self.config.joint_c]
        return _angle_deg(a, b, c)

    def update(self, lm: Optional[PoseLandmarks]) -> tuple[int, dict]:
        """输入当前帧的关键点，返回(累计计数, 诊断信息)。
        lm 为空或不可见则保持状态。
        """
        self.frame_idx += 1
        info = {"state": self._state, "angle": float("nan")}

        if lm is None or not self._visible_ok(lm):
            return self.count, info

        angle = self._current_angle(lm)
        info["angle"] = angle
        if not np.isfinite(angle):
            return self.count, info

        # 限制频繁切换
        can_switch = (self.frame_idx - self._last_switch_frame_idx) >= self.config.min_frames_between

        if angle >= self.config.angle_high_deg:
            if can_switch and self._state == "low":
                # 完成一次低->高，若之前经历了高->低，则计数+1
                self.count += 1
                self._last_switch_frame_idx = self.frame_idx
            self._state = "high"
        elif angle <= self.config.angle_low_deg:
            if can_switch and self._state == "high":
                self._last_switch_frame_idx = self.frame_idx
            self._state = "low"
        # 介于中间不改变状态

        info["state"] = self._state
        return self.count, info

    def set_config(self, **kwargs) -> None:
        """动态调整配置，如 joint 索引或阈值。"""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)


@dataclass
class MultiCounterConfig:
    triplets: List[Tuple[int, int, int]]
    angle_low_deg: float = 50.0
    angle_high_deg: float = 160.0
    min_visible: float = 0.5
    min_frames_between: int = 3


class MultiRepetitionCounter:
    """多特征重复计数器：同时跟踪多个关节三元组角度，
    通过多数投票的高/低状态切换完成一次重复统计。"""

    def __init__(self, config: MultiCounterConfig) -> None:
        self.config = config
        self._state = "unknown"  # "high" | "low" | "unknown"
        self._last_switch_frame_idx = -999
        self.count = 0
        self.frame_idx = 0

    def reset(self) -> None:
        self._state = "unknown"
        self._last_switch_frame_idx = -999
        self.count = 0
        self.frame_idx = 0

    def _visible_ok_triplet(self, lm: PoseLandmarks, tri: Tuple[int, int, int]) -> bool:
        a, b, c = tri
        vis = lm.data[[a, b, c], 3]
        return bool(np.all(vis >= self.config.min_visible))

    def _angle_for_triplet(self, lm: PoseLandmarks, tri: Tuple[int, int, int]) -> float:
        a, b, c = tri
        return _angle_deg(lm.xy[a], lm.xy[b], lm.xy[c])

    def update(self, lm: Optional[PoseLandmarks]) -> tuple[int, dict]:
        self.frame_idx += 1
        info = {"state": self._state, "angles": [], "per_state": []}
        if lm is None:
            return self.count, info
        angles: list[float] = []
        states: list[str] = []
        for tri in self.config.triplets:
            if not self._visible_ok_triplet(lm, tri):
                angles.append(float("nan"))
                states.append("unknown")
                continue
            ang = self._angle_for_triplet(lm, tri)
            angles.append(ang)
            if not np.isfinite(ang):
                states.append("unknown")
            elif ang >= self.config.angle_high_deg:
                states.append("high")
            elif ang <= self.config.angle_low_deg:
                states.append("low")
            else:
                states.append("mid")
        info["angles"] = angles
        info["per_state"] = states

        # 多数投票得到总体状态（忽略 unknown/mid，若都为 mid/unknown 则保持）
        high_cnt = states.count("high")
        low_cnt = states.count("low")
        new_state = self._state
        if high_cnt > low_cnt and high_cnt > 0:
            new_state = "high"
        elif low_cnt > high_cnt and low_cnt > 0:
            new_state = "low"
        # 限制频繁切换
        can_switch = (self.frame_idx - self._last_switch_frame_idx) >= self.config.min_frames_between
        if new_state == "high" and self._state == "low" and can_switch:
            self.count += 1
            self._last_switch_frame_idx = self.frame_idx
        if new_state in ("high", "low"):
            self._state = new_state
        info["state"] = self._state
        return self.count, info


@dataclass
class CompositeCounterConfig:
    triplets: List[Tuple[int, int, int]]  # 多关节集合
    angle_low_deg: float = 50.0
    angle_high_deg: float = 160.0
    min_visible: float = 0.5
    min_frames_between: int = 3
    # 阶段窗口：避免抖动，单次重复最长帧数限制（0表示不限制）
    max_frames_per_rep: int = 0


class CompositeRepetitionCounter:
    """复合模式计数器：通过多关节联合条件驱动阶段机，统计完整重复。
    示例策略（适用于深蹲/波比跳等基础变体）：
    - down 阶段：腿与臂均满足低阈（收缩）为主。
    - up 阶段：腿与臂均满足高阈（伸展）为主。
    完整重复：down -> up。
    注：该策略比简单多数投票更严格，需多关节同步满足条件。"""

    def __init__(self, config: CompositeCounterConfig) -> None:
        self.config = config
        self._state = "unknown"  # "down" | "up" | "unknown"
        self._last_switch_frame_idx = -999
        self.count = 0
        self.frame_idx = 0
        self._rep_start_frame = None

    def reset(self) -> None:
        self._state = "unknown"
        self._last_switch_frame_idx = -999
        self.count = 0
        self.frame_idx = 0
        self._rep_start_frame = None

    def _visible_ok_triplet(self, lm: PoseLandmarks, tri: Tuple[int, int, int]) -> bool:
        a, b, c = tri
        vis = lm.data[[a, b, c], 3]
        return bool(np.all(vis >= self.config.min_visible))

    def _angle_for_triplet(self, lm: PoseLandmarks, tri: Tuple[int, int, int]) -> float:
        a, b, c = tri
        return _angle_deg(lm.xy[a], lm.xy[b], lm.xy[c])

    def _all_angles_and_states(self, lm: PoseLandmarks) -> tuple[List[float], List[str]]:
        angles: list[float] = []
        states: list[str] = []
        for tri in self.config.triplets:
            if not self._visible_ok_triplet(lm, tri):
                angles.append(float("nan"))
                states.append("unknown")
                continue
            ang = self._angle_for_triplet(lm, tri)
            angles.append(ang)
            if not np.isfinite(ang):
                states.append("unknown")
            elif ang >= self.config.angle_high_deg:
                states.append("high")
            elif ang <= self.config.angle_low_deg:
                states.append("low")
            else:
                states.append("mid")
        return angles, states

    def update(self, lm: Optional[PoseLandmarks]) -> tuple[int, dict]:
        self.frame_idx += 1
        info = {"state": self._state, "angles": [], "per_state": []}
        if lm is None:
            return self.count, info
        angles, states = self._all_angles_and_states(lm)
        info["angles"] = angles
        info["per_state"] = states

        # 联合条件：选定集合（默认使用前两项臂 + 后两项腿）
        n = len(states)
        arm_states = states[:2] if n >= 2 else states
        leg_states = states[2:4] if n >= 4 else states[2:]
        arms_low = all(s == "low" for s in arm_states if s != "unknown") and any(s == "low" for s in arm_states)
        legs_low = all(s == "low" for s in leg_states if s != "unknown") and any(s == "low" for s in leg_states)
        arms_high = all(s == "high" for s in arm_states if s != "unknown") and any(s == "high" for s in arm_states)
        legs_high = all(s == "high" for s in leg_states if s != "unknown") and any(s == "high" for s in leg_states)

        # 可切换判定
        can_switch = (self.frame_idx - self._last_switch_frame_idx) >= self.config.min_frames_between

        # down: 同时满足臂低与腿低
        if arms_low and legs_low:
            if self._state != "down" and can_switch:
                self._state = "down"
                self._last_switch_frame_idx = self.frame_idx
                self._rep_start_frame = self.frame_idx
        # up: 同时满足臂高与腿高
        elif arms_high and legs_high:
            if self._state == "down" and can_switch:
                # 完成 down->up
                self.count += 1
                self._state = "up"
                self._last_switch_frame_idx = self.frame_idx
                self._rep_start_frame = None
            elif self._state != "up" and can_switch:
                self._state = "up"
                self._last_switch_frame_idx = self.frame_idx
        else:
            # 其他状态保持
            pass

        # 超时保护：单次重复过长则重置状态
        if self.config.max_frames_per_rep and self._rep_start_frame is not None:
            if (self.frame_idx - self._rep_start_frame) > self.config.max_frames_per_rep:
                self._state = "unknown"
                self._rep_start_frame = None

        info["state"] = self._state
        return self.count, info


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
        否则：以段长 seg_len=T/key_actions，将上限设置为到下一个关键动作中心的距离。
        """
        if not self.key_actions:
            return int(max(0, self.cfg.lookahead))
        if self.key_actions <= 0 or self.T <= 0:
            return int(max(0, self.cfg.lookahead))
        seg_len = float(self.T) / float(self.key_actions)
        # 当前所属关键动作段序号
        seg_idx = int(self.idx // seg_len)
        # 下一个关键动作中心索引 ≈ (seg_idx+1+0.5)*seg_len
        center_next = int(round((seg_idx + 1.5) * seg_len))
        # 不能超过模板末尾
        center_next = min(self.T - 1, max(0, center_next))
        return max(0, center_next - self.idx)

    def update(self, lm: Optional[PoseLandmarks]) -> tuple[int, dict]:
        info = {"idx": self.idx, "T": self.T, "passed": False, "diffs": None, "skipped": 0}
        if lm is None:
            return self.count, info
        # 当前帧角度
        angs = compute_joint_angles(lm.data, visibility_threshold=self.cfg.visibility_th)
        user_vec = np.array([float(angs.get(k, np.nan)) for k in self.template.keys], dtype=np.float32)

        # 尝试匹配 当前模板帧 -> 向后若干帧（动态前瞻）
        def match_at(t_idx: int) -> tuple[bool, Optional[np.ndarray]]:
            ref_vec = self.template.angles[t_idx]
            ref_valid = self.template.valid[t_idx]
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
            max_look = int(min(self._dynamic_lookahead(), max(0, self.T - self.idx - 1)))
            for j in range(1, max_look + 1):
                ok_j, diffs_j = match_at(self.idx + j)
                if ok_j:
                    matched = True
                    adv = j + 1
                    info["diffs"] = diffs_j
                    info["skipped"] = j
                    break
        if matched:
            self.idx += adv
            if self.idx >= self.T:
                self.count += 1
                self.idx = 0
            info["passed"] = True
        info["idx"] = self.idx
        return self.count, info
