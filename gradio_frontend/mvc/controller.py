from __future__ import annotations

import os
from typing import Optional, Generator, Tuple
import cv2
import numpy as np

from .model import (
    save_uploaded,
    _draw_annotations,
)

from dance_core.motion_counter import AngleTemplate, TemplateRepetitionCounter, TemplateMatcherConfig
from dance_core.pose_detector import PoseDetector, PoseDetectorConfig
from dance_core.types import PoseLandmarks
from dance_core.scoring import compute_joint_angles

DEFAULT_MIN_VISIBLE = 0.5
DEFAULT_MIN_GAP_FRAMES = 3
DEFAULT_ANGLE_LOW = 50.0
DEFAULT_ANGLE_HIGH = 160.0


class CountController:
    """业务控制器，负责参数组织与流程调度。"""

    def __init__(self) -> None:
        # 保存参考视频分析得到的参数
        self.ref_triplet: Optional[Tuple[int, int, int]] = None
        self.ref_angle_low: float = DEFAULT_ANGLE_LOW
        self.ref_angle_high: float = DEFAULT_ANGLE_HIGH
        self.min_visible: float = DEFAULT_MIN_VISIBLE
        self.min_gap_frames: int = DEFAULT_MIN_GAP_FRAMES
        # 参考模板（逐帧多关节角度）
        self.ref_template: Optional[AngleTemplate] = None
        # 最近一次设置的关键动作数（用于渲染阶段沿用）
        self.last_key_actions: Optional[int] = None

    # 保留但不再使用旧的简化接口
    def start_simple(
        self,
        ref_file: Optional[str],
        eval_file: Optional[str],
    ) -> Generator[tuple, None, None]:
        import numpy as np
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        yield ("该接口已弃用，请按两阶段流程使用界面按钮", "", blank, blank)

    def start_reference(self, ref_file: Optional[str]) -> Generator[tuple, None, None]:
        """先处理参考视频：校准阈值并构建模板，同时播放参考视频标注预览。"""
        ref_path = save_uploaded(ref_file, "ref")
        if not ref_path:
            import numpy as np
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("未提供参考视频", blank)
            return
        # 参考阈值可保留（用于其他展示），但计数完全采用模板
        self.ref_angle_low, self.ref_angle_high = self.ref_angle_low, self.ref_angle_high
        # 构建模板（10fps）
        try:
            self.ref_template = AngleTemplate.from_video(ref_path, sample_fps=10.0, visibility_th=self.min_visible)
            tmpl_len = self.ref_template.angles.shape[0]
            tmpl_info = f"模板帧数: {tmpl_len}"
        except Exception as e:
            self.ref_template = None
            tmpl_info = f"模板构建失败: {e}"
        # 播放参考视频预览（仅绘制，不计数）
        pd = PoseDetector(PoseDetectorConfig())
        cap = cv2.VideoCapture(ref_path)
        import numpy as np
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
                pass

    def start_evaluation(self, eval_file: Optional[str], use_webcam: bool = False) -> Generator[tuple, None, None]:
        """在已完成参考处理后，开始评测：若存在参考模板则优先使用模板匹配计数。"""
        if self.ref_template is not None:
            # 走模板评测
            yield from self.start_template_evaluation(eval_file, use_webcam=use_webcam, tolerance_deg=10.0)
            return
        # 否则提示
        import numpy as np
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        yield ("未构建参考模板，请先在步骤1处理参考视频", blank)

    def start_template_evaluation(self, eval_file: Optional[str], use_webcam: bool = False, tolerance_deg: float = 10.0, key_actions: Optional[int] = None) -> Generator[tuple, None, None]:
        """基于参考模板逐帧匹配的评测：每帧各关节角度与模板差均小于容差则推进，模板走完计一次。支持按关键动作数动态前瞻。"""
        if self.ref_template is None:
            import numpy as np
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            yield ("请先在步骤1处理参考视频以构建模板", blank)
            return
        # 保存关键动作数以便渲染沿用
        self.last_key_actions = key_actions if (key_actions and key_actions > 0) else None
        eval_path = 0 if use_webcam else save_uploaded(eval_file, "eval")
        # 准备检测器与计数器
        pd = PoseDetector(PoseDetectorConfig())
        matcher = TemplateRepetitionCounter(
            self.ref_template,
            TemplateMatcherConfig(tolerance_deg=tolerance_deg, visibility_th=self.min_visible),
            key_actions=self.last_key_actions,
        )
        cap = None if eval_path is None else cv2.VideoCapture(eval_path)
        while True:
            if not cap or not cap.isOpened():
                break
            ok, frame = cap.read()
            if not ok:
                break
            lm = pd.detect_landmarks(frame)
            plm = PoseLandmarks(lm) if lm is not None else None
            cnt, info = matcher.update(plm)
            # 取主角度显示（右臂角度）
            main_angle = None
            import numpy as np
            if lm is not None:
                angs = compute_joint_angles(lm, visibility_threshold=self.min_visible)
                a0 = float(angs.get("r_elbow", float("nan")))
                main_angle = a0 if np.isfinite(a0) else None
            ann = _draw_annotations(frame, plm, (12, 14, 16), main_angle)
            txt = f"模板进度: {info.get('idx')}/{info.get('T')} | 已完成: {cnt} | 匹配:{'✓' if info.get('passed') else '×'} | 前瞻跳过:{info.get('skipped')}"
            yield (txt, ann)
        try:
            if cap:
                cap.release()
            pd.close()
        except Exception:
            pass

    def render_session(self, ref_file: Optional[str], eval_file: Optional[str]) -> str:
        """渲染整段会话视频：评测为主画面，参考画面缩小为画中画；计数采用模板匹配。"""
        ref_path = save_uploaded(ref_file, "ref")
        eval_path = save_uploaded(eval_file, "eval")
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".temp", "session_output.mp4")

        # 确保模板可用
        if self.ref_template is None and ref_path:
            try:
                self.ref_template = AngleTemplate.from_video(ref_path, sample_fps=10.0, visibility_th=self.min_visible)
            except Exception:
                self.ref_template = None

        # 初始化检测与计数器（沿用最近的关键动作数）
        pd = PoseDetector(PoseDetectorConfig())
        matcher = TemplateRepetitionCounter(
            self.ref_template,
            TemplateMatcherConfig(tolerance_deg=10.0, visibility_th=self.min_visible),
            key_actions=self.last_key_actions,
        ) if self.ref_template else None

        cap_ref = cv2.VideoCapture(ref_path) if ref_path else None
        cap_eval = cv2.VideoCapture(eval_path) if eval_path else None

        def cap_size(cap):
            if cap is not None and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                return w, h, fps
            return None
        sz_e = cap_size(cap_eval)
        sz_r = cap_size(cap_ref)
        base_w, base_h, base_fps = (sz_e or sz_r or (640, 480, 25.0))

        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_fn is None:
            raise RuntimeError("OpenCV 缺少 VideoWriter_fourcc")
        fourcc = fourcc_fn(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, base_fps, (base_w, base_h))

        try:
            while True:
                progressed = False
                frame_r = None
                frame_e = None
                if cap_ref and cap_ref.isOpened():
                    ok_r, fr = cap_ref.read()
                    if ok_r:
                        progressed = True
                        frame_r = fr
                if cap_eval and cap_eval.isOpened():
                    ok_e, fe = cap_eval.read()
                    if ok_e:
                        progressed = True
                        frame_e = fe
                if not progressed:
                    break

                base = frame_e if frame_e is not None else frame_r
                if base is None:
                    break
                vis = base.copy()

                # 评测绘制与计数
                lm_e = pd.detect_landmarks(frame_e) if frame_e is not None else None
                plm_e = PoseLandmarks(lm_e) if lm_e is not None else None
                cnt = 0
                idx = 0
                passed = False
                if matcher is not None:
                    cnt, info = matcher.update(plm_e)
                    idx = int(info.get("idx", 0))
                    passed = bool(info.get("passed", False))
                main_angle = None
                if lm_e is not None:
                    angs = compute_joint_angles(lm_e, visibility_threshold=self.min_visible)
                    a0 = float(angs.get("r_elbow", float("nan")))
                    main_angle = a0 if np.isfinite(a0) else None
                vis = _draw_annotations(vis, plm_e, (12, 14, 16), main_angle)
                cv2.putText(vis, f"已完成:{cnt} 模板进度:{idx}{' ✓' if passed else ''}", (20, base_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                # 参考画面画中画
                if frame_r is not None:
                    lm_r = pd.detect_landmarks(frame_r)
                    plm_r = PoseLandmarks(lm_r) if lm_r is not None else None
                    ang_ref = None
                    if lm_r is not None:
                        angs_r = compute_joint_angles(lm_r, visibility_threshold=self.min_visible)
                        a0r = float(angs_r.get("r_elbow", float("nan")))
                        ang_ref = a0r if np.isfinite(a0r) else None
                    pip = _draw_annotations(frame_r, plm_r, (12, 14, 16), ang_ref)
                    rw = int(base_w * 0.3)
                    rh = int(base_h * 0.3)
                    pip = cv2.resize(pip, (rw, rh))
                    vis[0:rh, 0:rw] = pip
                    cv2.rectangle(vis, (0,0), (rw-1,rh-1), (0,0,0), 2)

                writer.write(vis)
        finally:
            try:
                writer.release()
                if cap_ref:
                    cap_ref.release()
                if cap_eval:
                    cap_eval.release()
                pd.close()
            except Exception:
                pass

        return out_path

