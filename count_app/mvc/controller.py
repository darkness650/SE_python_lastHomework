"""
计数控制器 - 基于原有Gradio MVC controller
"""
from __future__ import annotations

import os
from typing import Optional, Generator, Tuple
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
    """业务控制器，负责参数组织与流程调度。"""

    def __init__(self) -> None:
        # 可见度阈值
        self.min_visible: float = DEFAULT_MIN_VISIBLE
        # 参考模板（逐帧多关节角度）
        self.ref_template: Optional[AngleTemplate] = None
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

    def start_reference(self, ref_file: Optional[str]) -> Generator[Tuple[str, np.ndarray], None, None]:
        """先处理参考视频：校准阈值并构建模板，同时播放参考视频标注预览。"""
        ref_path = save_uploaded(ref_file, "ref")
        if not ref_path:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("未提供参考视频", blank)
            return

        # 构建模板（使用更高的采样率）
        try:
            self.ref_template = AngleTemplate.from_video(
                ref_path,
                sample_fps=25.0,  # 提高采样率
                visibility_th=self.min_visible
            )
            tmpl_len = self.ref_template.angles.shape[0]
            # 记录参考抽样FPS
            self.ref_sample_fps = 25.0
            tmpl_info = f"模板帧数: {tmpl_len} | 抽样FPS:  {self.ref_sample_fps}"
        except Exception as e:
            self.ref_template = None
            self.ref_sample_fps = None
            tmpl_info = f"模板构建失败: {e}"

        # 播放参考视频预览（仅绘制，不计数）
        pd = PoseDetector(PoseDetectorConfig())
        cap = cv2.VideoCapture(ref_path)

        # 设置更高分辨率
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while cap.isOpened():
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
                cap.release()
                pd.close()
            except Exception:
                print("error")

    def start_template_evaluation(self, eval_file: Optional[str], use_webcam: bool = False,
                                  tolerance_deg: float = 10.0, key_actions: Optional[int] = None) -> Generator[
        Tuple[str, np.ndarray], None, None]:
        """基于参考模板逐帧匹配的评测"""
        if self.ref_template is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("请先在步骤1处理参考视频以构建模板", blank)
            return

        # 保存关键动作数
        self.last_key_actions = key_actions if key_actions is not None else None
        # 重置停止标志
        self._stop = False
        eval_path = 0 if use_webcam else save_uploaded(eval_file, "eval")

        # 准备检测器与计数器
        pd = PoseDetector(PoseDetectorConfig())
        matcher_cfg = TemplateMatcherConfig(tolerance_deg=tolerance_deg, visibility_th=self.min_visible)
        if key_actions == 0:
            matcher_cfg.lookahead = 4
        matcher = TemplateRepetitionCounter(
            self.ref_template,
            matcher_cfg,
            key_actions=self.last_key_actions,
        )

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

        # 初始化总计与最后一帧标注
        total_cnt = 0
        last_ann = None

        try:
            while True:
                # 手动停止优先
                if self._stop:
                    break
                if not cap or not cap.isOpened():
                    # 摄像头模式下，若暂时不可读，稍作等待继续尝试
                    if use_webcam:
                        time.sleep(0.02)
                        continue
                    break

                ok, frame = cap.read()
                if not ok:
                    # 摄像头模式下继续尝试读取
                    if use_webcam:
                        time.sleep(0.02)
                        continue
                    break

                lm = pd.detect_landmarks(frame)
                plm = PoseLandmarks(lm) if lm is not None else None
                cnt, info = matcher.update(plm)
                total_cnt = cnt

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

                # 文本包含FPS
                txt = f"模板进度: {info.get('idx')}/{info.get('T')} | 已完成: {cnt} | 匹配: {'✓' if info.get('passed') else '×'} | 前瞻跳过:{info.get('skipped')} | 处理FPS:{proc_fps:.1f}"
                yield (txt, ann)

                # 节流：尽量让处理速度与源播放速度一致
                target_fps = src_fps if src_fps > 0 else proc_fps
                if target_fps > 0:
                    frame_period = 1.0 / target_fps
                    # 避免过长睡眠，保持交互流畅
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
            final_txt = f"处理结束，总计数一共 {total_cnt}"
            yield (final_txt, last_ann)