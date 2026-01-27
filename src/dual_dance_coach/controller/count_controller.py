"""计数控制器"""

import time
from typing import Generator

import cv2
import numpy as np

from dual_dance_coach.core.motion_counter import (
    AngleTemplate,
    TemplateMatcherConfig,
    TemplateRepetitionCounter,
)
from dual_dance_coach.core.pose_connections import POSE_CONNECTIONS
from dual_dance_coach.core.pose_detector import PoseDetector, PoseDetectorConfig
from dual_dance_coach.core.scoring import compute_joint_angles
from dual_dance_coach.core.types import PoseLandmarks

DEFAULT_MIN_VISIBLE = 0.5


def draw_annotations(
    frame: np.ndarray,
    landmarks: PoseLandmarks | None,
    triplet: tuple[int, int, int],
    angle: float | None,
) -> np.ndarray:
    """在帧上绘制关键点、全身骨架与角度文字（包含多关节角度）。"""
    # 不要改成 or
    if landmarks is None and angle is None:
        return frame

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
        # 画全身骨架 - 使用抗锯齿
        for a, b in POSE_CONNECTIONS:
            try:
                if landmarks.data[a, 3] >= 0 and landmarks.data[b, 3] >= 0:
                    cv2.line(vis, to_xy(a), to_xy(b), color_skeleton, 3, cv2.LINE_AA)
            except Exception as e:
                print(e)
        # 高亮当前三元组关键点与连线
        a, b, c = triplet
        for idx in [a, b, c]:
            if landmarks.data[idx, 3] >= 0:
                x, y = to_xy(idx)
                cv2.circle(vis, (x, y), 8, color_pts, -1)
        try:
            cv2.line(vis, to_xy(a), to_xy(b), color_line, thickness + 1, cv2.LINE_AA)
            cv2.line(vis, to_xy(b), to_xy(c), color_line, thickness + 1, cv2.LINE_AA)
        except Exception as e:
            print(e)

    # 角度文字：主角度 + 多关节角度（上肢与下肢）
    # 中文不行，用英文代替
    text_lines = []
    # text_lines.append(f"主角度: {angle:.1f}°" if angle is not None else "主角度: N/A")
    text_lines.append(f"Main Angle: {angle:.1f}°" if angle is not None else "Main Angle: N/A")
    if landmarks is not None:
        # 定义多关节三元组：双臂与双腿
        multi_triplets = [
            (12, 14, 16),  # 右臂
            (11, 13, 15),  # 左臂
            (24, 26, 28),  # 右腿
            (23, 25, 27),  # 左腿
        ]
        # names = ["右臂", "左臂", "右腿", "左腿"]
        names = ["Right Arm", "Left Arm", "Right Leg", "Left Leg"]
        for name, tri in zip(names, multi_triplets):
            ang = _compute_angle_deg(landmarks, *tri)
            text_lines.append(f"{name}: {ang:.1f}" if ang is not None else f"{name}: N/A")

    # 绘制文字块 - 增加背景和更大字体
    y0 = 40
    font_scale = 0.9
    for i, t in enumerate(text_lines):
        # 计算文字大小
        (text_width, text_height), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        # 绘制背景
        cv2.rectangle(
            vis,
            (15, y0 + i * 35 - text_height - 5),
            (25 + text_width, y0 + i * 35 + 5),
            (0, 0, 0),
            -1,
        )
        # 绘制文字
        cv2.putText(
            vis,
            t,
            (20, y0 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_text,
            2,
            cv2.LINE_AA,
        )
    return vis


class CountController:
    """业务控制器，负责参数组织与流程调度。支持多标准视频与自动切换动作。"""

    def __init__(self) -> None:
        # 可见度阈值
        self.min_visible: float = DEFAULT_MIN_VISIBLE
        # 单一或多参考模板（逐帧多关节角度）
        self.ref_template: AngleTemplate | None = None
        self.ref_templates: list[AngleTemplate] = []
        # 每个动作的目标次数（与模板对齐后的版本）
        self.action_targets: list[int] = []
        # 原始用户设置的目标次数（可能在构建模板之前设置）
        self._raw_action_targets: list[int] = []
        # 最近一次设置的关键动作数（用于渲染阶段沿用）
        self.last_key_actions: int | None = None
        # 记录处理FPS
        self.ref_sample_fps: float | None = None
        self.last_proc_fps: float | None = None
        # 停止标志（用于摄像头模式手动停止）
        self._stop: bool = False

    def stop(self) -> None:
        """请求停止当前评测（主要用于摄像头实时模式）。"""
        self._stop = True

    def start_reference(
        self, ref_files: list[str]
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        处理参考视频：支持单个或多个。构建模板并播放第一个参考视频的标注预览。
        如果传入列表，则逐个构建模板，并记录至 self.ref_templates 与 self.action_targets（目标次数由UI设置）。
        """
        if not ref_files:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("未提供参考视频", blank)
            return

        # 清空旧模板
        self.ref_templates = []

        # 逐个构建模板
        build_msgs: list[str] = []
        for idx, rf in enumerate(ref_files, start=1):
            try:
                tmpl = AngleTemplate.from_video(
                    rf,
                    sample_fps=25.0,
                    visibility_th=self.min_visible,
                )
                self.ref_templates.append(tmpl)
                build_msgs.append(f"模板{idx}构建成功: 帧数 {tmpl.angles.shape[0]}")
            except Exception as e:
                build_msgs.append(f"模板{idx}构建失败: {e}")
        # 记录参考抽样FPS
        self.ref_sample_fps = 25.0

        # 构建完成后，根据原始目标次数重新对齐 action_targets
        self._recompute_action_targets()

        if not self.ref_templates:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("模板构建失败：未能生成任何模板\n" + "\n".join(build_msgs), blank)
            return

        # 兼容旧逻辑：将第一个模板作为 ref_template
        self.ref_template = self.ref_templates[0]
        tmpl_info = (
            "\n".join(build_msgs)
            + f"\n共构建模板 {len(self.ref_templates)} 个 | 抽样FPS: {self.ref_sample_fps}"
        )

        # 参考预览：仅播放第一个参考视频（若可用）
        pd = PoseDetector(PoseDetectorConfig())
        first_path = ref_files[0]
        cap = cv2.VideoCapture(first_path) if first_path else None

        # 设置更高分辨率
        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while cap and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                lm = pd.detect_landmarks(frame)
                plm = PoseLandmarks(lm) if lm is not None else None
                main_angle = None
                if lm is not None:
                    angs = compute_joint_angles(lm, visibility_threshold=self.min_visible)
                    a0 = float(angs.get("r_elbow", float("nan")))
                    main_angle = a0 if np.isfinite(a0) else None
                ann = draw_annotations(frame, plm, (12, 14, 16), main_angle)
                yield (tmpl_info, ann)
        finally:
            try:
                if cap:
                    cap.release()
                pd.close()
            except Exception as e:
                print(e)

    def set_action_targets(self, targets: list[int]) -> None:
        """设置每个动作的目标次数，与 ref_templates 一一对应。长度不匹配时按最小长度截断。
        支持在模板构建前或后调用：会保存原始列表并在需要时进行对齐。"""
        self._raw_action_targets = [
            max(0, int(t)) for t in targets if isinstance(t, (int, float, str))
        ]
        self._recompute_action_targets()

    def _recompute_action_targets(self) -> None:
        """根据当前模板数量，对齐并更新实际使用的 action_targets。"""
        if not self.ref_templates:
            # 模板尚未构建，暂不对齐，保留空以避免误用
            self.action_targets = []
            return
        n = min(len(self.ref_templates), len(self._raw_action_targets))
        self.action_targets = self._raw_action_targets[:n]

    def start_template_evaluation(
        self,
        eval_file: str | None,
        use_webcam: bool = False,
        tolerance_deg: float = 10.0,
        key_actions: int | None = None,
    ) -> Generator[tuple[str, np.ndarray | None], None, None]:
        """基于参考模板逐帧匹配的评测。支持按目标次数自动切换至下一个模板，直到所有动作完成。"""
        # 至少需要一个模板
        active_templates: list[AngleTemplate] = (
            self.ref_templates
            if self.ref_templates
            else ([] if self.ref_template is None else [self.ref_template])
        )
        if not active_templates:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("请先在步骤1处理参考视频以构建模板", blank)
            return

        # 确保 action_targets 已与模板对齐（防止用户在构建前就设置了targets）
        self._recompute_action_targets()

        # 保存关键动作数
        self.last_key_actions = key_actions if key_actions is not None else None
        # 重置停止标志
        self._stop = False

        eval_path = None if use_webcam else eval_file

        # 准备检测器
        pd = PoseDetector(PoseDetectorConfig())
        # 当前动作索引
        action_idx = 0
        # 每个动作已完成次数
        reps_done = 0

        def make_matcher(tmpl: AngleTemplate) -> TemplateRepetitionCounter:
            cfg = TemplateMatcherConfig(tolerance_deg=tolerance_deg, visibility_th=self.min_visible)
            if self.last_key_actions == 0:
                cfg.lookahead = 4
            return TemplateRepetitionCounter(
                tmpl,
                cfg,
                key_actions=self.last_key_actions,
            )

        matcher = make_matcher(active_templates[action_idx])

        cap = None
        # 摄像头模式
        if use_webcam:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
        # 文件模式
        else:
            if eval_path is not None:
                cap = cv2.VideoCapture(eval_path)

        # 处理FPS统计
        t0 = time.time()
        frames_done = 0

        # 若有源视频FPS，作为参考播放速度
        src_fps = 0.0
        if cap and cap.isOpened():
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        last_ann = None

        try:
            while True:
                if self._stop:
                    break
                if not cap or not cap.isOpened():
                    if use_webcam:
                        time.sleep(0.02)
                        continue
                    break
                ok, frame = cap.read()
                if not ok:
                    if use_webcam:
                        time.sleep(0.02)
                        continue
                    break

                lm = pd.detect_landmarks(frame)
                plm = PoseLandmarks(lm) if lm is not None else None
                cnt, info = matcher.update(plm)
                reps_done = cnt

                # 目标次数阈值（缺省或为0时默认1次，保证可以前进）
                raw_target = (
                    self.action_targets[action_idx]
                    if action_idx < len(self.action_targets)
                    else None
                )
                target = raw_target if (raw_target is not None and raw_target > 0) else 1
                reached = reps_done >= target

                # 取主角度显示（右臂角度）
                main_angle = None
                if lm is not None:
                    angs = compute_joint_angles(lm, visibility_threshold=self.min_visible)
                    a0 = float(angs.get("r_elbow", float("nan")))
                    main_angle = a0 if np.isfinite(a0) else None

                ann = draw_annotations(frame, plm, (12, 14, 16), main_angle)
                last_ann = ann

                # 更新处理FPS
                frames_done += 1
                elapsed = time.time() - t0
                proc_fps = frames_done / elapsed if elapsed > 0 else 0.0
                self.last_proc_fps = proc_fps

                # 文本包含动作索引、目标与FPS
                action_txt = f"动作 {action_idx + 1}/{len(active_templates)}"
                target_txt = f" | 目标:{target}"
                txt = f"{action_txt}{target_txt} | 模板进度: {info.get('idx')}/{info.get('T')} | 已完成: {reps_done} | 匹配: {'✓' if info.get('passed') else '×'} | 前瞻跳过:{info.get('skipped')} | 处理FPS:{proc_fps:.1f}"
                yield (txt, ann)

                # 自动切换到下一个动作
                if reached:
                    action_idx += 1
                    if action_idx >= len(active_templates):
                        break
                    # 记录切换提示
                    yield (f"切换到动作 {action_idx + 1}", ann)
                    matcher = make_matcher(active_templates[action_idx])
                    reps_done = 0
                    t0 = time.time()
                    frames_done = 0

                # 节流：尽量让处理速度与源播放速度一致
                target_fps = src_fps if src_fps > 0 else proc_fps
                if target_fps > 0:
                    frame_period = 1.0 / target_fps
                    time.sleep(min(frame_period, 0.05))

        finally:
            try:
                if cap:
                    cap.release()
                pd.close()
            except Exception as e:
                print(e)

        # 文件模式结束后输出最终总计
        if not use_webcam:
            final_txt = f"评审结束，已完成 {len(active_templates)} 个动作序列"
            yield (final_txt, last_ann)


def _compute_angle_deg(landmarks: PoseLandmarks, a: int, b: int, c: int) -> float | None:
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
