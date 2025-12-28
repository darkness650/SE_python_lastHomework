"""
计数控制器 - 基于原有Gradio MVC controller（扩展为多标准视频与目标次数）
"""
from __future__ import annotations

import os
from typing import Optional, Generator, Tuple, List
import cv2
import numpy as np

# 导入原有的模型层函数
from .model import (
    save_uploaded,
    _draw_annotations,
)

# 导入原有的核心模块
from dance_core.motion_counter import AngleTemplate, TemplateRepetitionCounter, TemplateMatcherConfig
from dance_core.pose_detector import PoseDetector, PoseDetectorConfig
from dance_core.types import PoseLandmarks
from dance_core.scoring import compute_joint_angles

DEFAULT_MIN_VISIBLE = 0.5


class CountController:
    """业务控制器，负责参数组织与流程调度。支持多标准视频与自动切换动作。"""

    def __init__(self) -> None:
        # 可见度阈值
        self.min_visible: float = DEFAULT_MIN_VISIBLE
        # 单一或多参考模板（逐帧多关节角度）
        self.ref_template: Optional[AngleTemplate] = None
        self.ref_templates: List[AngleTemplate] = []
        # 每个动作的目标次数（与模板对齐后的版本）
        self.action_targets: List[int] = []
        # 原始用户设置的目标次数（可能在构建模板之前设置）
        self._raw_action_targets: List[int] = []
        # 最近一次设置的关键动作数（用于渲染阶段沿用）
        self.last_key_actions: Optional[int] = None
        # 记录处理FPS
        self.ref_sample_fps: Optional[float] = None
        self.last_proc_fps: Optional[float] = None
        # 停止标志（用于摄像头模式手动停止）
        self._stop: bool = False

    def stop(self) -> None:
        """请求停止当前评测（主要用于摄像头实时模式）。"""
        self._stop = True

    def start_reference(self, ref_file: Optional[str] | List[str]) -> Generator[Tuple[str, np.ndarray], None, None]:
        """处理参考视频：支持单个或多个。构建模板并播放第一个参考视频的标注预览。
        如果传入列表，则逐个构建模板，并记录至 self.ref_templates 与 self.action_targets（目标次数由UI设置）。
        """
        # 兼容旧接口：字符串或列表
        ref_files: List[str] = []
        if isinstance(ref_file, str) or ref_file is None:
            if ref_file:
                ref_files = [ref_file]
        elif isinstance(ref_file, list):
            ref_files = [p for p in ref_file if p]

        if not ref_files:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("未提供参考视频", blank)
            return

        # 清空旧模板
        self.ref_templates = []
        self.ref_template = None

        # 逐个构建模板
        build_msgs: List[str] = []
        for idx, rf in enumerate(ref_files, start=1):
            ref_path = save_uploaded(rf, f"ref{idx}")
            if not ref_path:
                build_msgs.append(f"参考视频{idx}保存失败")
                continue
            try:
                tmpl = AngleTemplate.from_video(
                    ref_path,
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
        tmpl_info = "\n".join(build_msgs) + f"\n共构建模板 {len(self.ref_templates)} 个 | 抽样FPS: {self.ref_sample_fps}"

        # 参考预览：仅播放第一个参考视频（若可用）
        pd = PoseDetector(PoseDetectorConfig())
        first_path = save_uploaded(ref_files[0], "ref_preview")
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
                ann = _draw_annotations(frame, plm, (12, 14, 16), main_angle)
                yield (tmpl_info, ann)
        finally:
            try:
                if cap:
                    cap.release()
                pd.close()
            except Exception:
                print("error")

    def set_action_targets(self, targets: List[int]) -> None:
        """设置每个动作的目标次数，与 ref_templates 一一对应。长度不匹配时按最小长度截断。
        支持在模板构建前或后调用：会保存原始列表并在需要时进行对齐。"""
        if not isinstance(targets, list):
            targets = []
        self._raw_action_targets = [max(0, int(t)) for t in targets if isinstance(t, (int, float, str))]
        self._recompute_action_targets()

    def _recompute_action_targets(self) -> None:
        """根据当前模板数量，对齐并更新实际使用的 action_targets。"""
        if not self.ref_templates:
            # 模板尚未构建，暂不对齐，保留空以避免误用
            self.action_targets = []
            return
        n = min(len(self.ref_templates), len(self._raw_action_targets))
        self.action_targets = self._raw_action_targets[:n]

    def start_template_evaluation(self, eval_file: Optional[str], use_webcam: bool = False,
                                  tolerance_deg: float = 10.0, key_actions: Optional[int] = None) -> Generator[
        Tuple[str, np.ndarray], None, None]:
        """基于参考模板逐帧匹配的评测。支持按目标次数自动切换至下一个模板，直到所有动作完成。"""
        # 至少需要一个模板
        active_templates: List[AngleTemplate] = self.ref_templates if self.ref_templates else ([] if self.ref_template is None else [self.ref_template])
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
        eval_path = 0 if use_webcam else save_uploaded(eval_file, "eval")

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

        cap = None if eval_path is None else cv2.VideoCapture(eval_path)

        # 设置高分辨率（对摄像头）
        if cap and cap.isOpened() and use_webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # 处理FPS统计
        import time
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
                raw_target = self.action_targets[action_idx] if action_idx < len(self.action_targets) else None
                target = raw_target if (raw_target is not None and raw_target > 0) else 1
                reached = reps_done >= target

                # 取主角度显示（右臂角度）
                main_angle = None
                if lm is not None:
                    angs = compute_joint_angles(lm, visibility_threshold=self.min_visible)
                    a0 = float(angs.get("r_elbow", float("nan")))
                    main_angle = a0 if np.isfinite(a0) else None

                ann = _draw_annotations(frame, plm, (12, 14, 16), main_angle)
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

