import time
from dataclasses import dataclass
from typing import TypeGuard

import cv2
import numpy as np
import shiboken6
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from dual_dance_coach.core.pose_connections import POSE_CONNECTIONS
from dual_dance_coach.core.pose_detector import PoseDetector
from dual_dance_coach.core.reference_extractor import (
    ExtractConfig,
    extract_reference_sequence,
    find_reference_by_time,
)
from dual_dance_coach.core.scoring import compare_angles
from dual_dance_coach.core.types import PoseFrame

from dual_dance_coach.view.dance.controller.view_protocol import PracticeView


def _bgr_to_qpixmap(frame_bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """BGR 帧转 QPixmap 并按最大尺寸等比缩放。

    输入: frame_bgr (h,w,3) BGR 图像；max_w/max_h 最大显示尺寸。
    输出: QPixmap。
    作用: 将 OpenCV 图像转换为可在 Qt 标签显示的位图。
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    pm = QPixmap.fromImage(qimg.copy())
    return pm.scaled(
        max_w,
        max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _draw_skeleton_bgr(frame_bgr: np.ndarray, landmarks_33x4: np.ndarray | None) -> np.ndarray:
    """在 BGR 帧上叠加骨架连线和关键点。

    输入: frame_bgr 原始图像；landmarks_33x4 为 (33,4) 或 None。
    输出: 带可视化叠加的图像副本。
    作用: 以可见度阈值 0.5 绘制骨架用于预览与反馈。
    """
    if landmarks_33x4 is None:
        return frame_bgr
    connections = POSE_CONNECTIONS

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    pts = []
    for i in range(33):
        x = int(float(landmarks_33x4[i, 0]) * w)
        y = int(float(landmarks_33x4[i, 1]) * h)
        v = float(landmarks_33x4[i, 3])
        pts.append((x, y, v))

    for a, b in connections:
        xa, ya, va = pts[a]
        xb, yb, vb = pts[b]
        if va < 0.5 or vb < 0.5:
            continue
        cv2.line(out, (xa, ya), (xb, yb), (0, 255, 0), 2)

    for x, y, v in pts:
        if v < 0.5:
            continue
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    return out


@dataclass
class RuntimeState:
    ref_video_path: str | None = None
    user_video_path: str | None = None
    reference_sequence: list[PoseFrame] | None = None
    running: bool = False
    start_time_s: float = 0.0


class ReferenceWorker(QObject):
    """后台线程工作者：抽取参考视频的姿态序列。

    输入: 构造时提供 video_path 与 sample_fps。
    输出: finished 信号发出 list[PoseFrame]；progress/failed 信号用于反馈。
    作用: 在线程中调用业务函数，避免阻塞 UI。
    """

    progress = Signal(int, int)
    finished = Signal(object)  # list[PoseFrame]
    failed = Signal(str)

    def __init__(self, video_path: str, sample_fps: float = 10.0):
        super().__init__()
        self._video_path = video_path
        self._sample_fps = sample_fps

    def run(self) -> None:
        """执行抽取流程：创建检测器、调用抽取、发出信号、清理资源。"""
        detector = None
        try:
            detector = PoseDetector()
            seq = extract_reference_sequence(
                self._video_path,
                detector,
                ExtractConfig(sample_fps=self._sample_fps),
                on_progress=lambda a, b: self.progress.emit(a, b),
            )
            self.finished.emit(seq)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if detector is not None:
                detector.close()


def _is_valid_thread(thread: QThread | None) -> TypeGuard[QThread]:
    """判断线程对象是否有效。

    输入: thread 可为 None。
    输出: True 表示可安全使用；否则为 False。
    作用: 结合 shiboken6 判断 Qt 线程对象生命周期状态。
    """
    try:
        return thread is not None and shiboken6.isValid(thread)  # type: ignore[attr-defined]
    except Exception:
        return False


class PracticeController(QObject):
    """控制器：承接UI事件，调用业务层；UI通过MainWindow暴露的接口更新。"""

    def __init__(self, view: PracticeView):
        """初始化控制器。

        输入: view 为实现 PracticeView 协议的视图对象。
        输出: 无。
        作用: 准备计时器、视频采集、姿态检测器与后台线程等运行资源。
        """
        super().__init__()
        self._view = view
        self._state = RuntimeState()

        self._ref_thread: QThread | None = None
        self._ref_worker: ReferenceWorker | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30fps
        self._timer.timeout.connect(self._on_tick)

        self._cap_user: cv2.VideoCapture | None = None
        self._cap_ref_preview: cv2.VideoCapture | None = None
        self._user_is_file: bool = False

        self._ref_preview_paused: bool = False
        self._ref_preview_speed: float = 1.0
        self._ref_preview_time_s: float = 0.0
        self._ref_preview_duration_s: float = 0.0
        self._ref_preview_last_ts: float = time.perf_counter()

        self._detector = PoseDetector()

        # 分数平滑（显示层面），避免检测噪声导致“横跳”
        self._score_ema: float | None = None
        self._score_alpha: float = 0.2
        self._score_sum: float = 0.0
        self._score_count: int = 0

    def load_reference(self, video_path: str) -> None:
        """加载并准备参考视频。

        输入: video_path 文件路径。
        输出: 无。
        作用: 停止当前练习、打开参考预览、启动后台抽取姿态序列。
        """
        self.stop()
        self._state.ref_video_path = video_path
        self._state.reference_sequence = None

        # 参考视频预览cap
        if self._cap_ref_preview is not None:
            self._cap_ref_preview.release()
        self._cap_ref_preview = cv2.VideoCapture(video_path)
        self._ref_preview_time_s = 0.0
        self._ref_preview_paused = False
        self._ref_preview_speed = 1.0
        self._ref_preview_last_ts = time.perf_counter()
        if self._cap_ref_preview is not None and self._cap_ref_preview.isOpened():
            fps = float(self._cap_ref_preview.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(self._cap_ref_preview.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            self._ref_preview_duration_s = frames / fps if fps > 0 else 0.0
        else:
            self._ref_preview_duration_s = 0.0

        # 后台抽取姿态序列
        self._start_reference_extraction(video_path)

    def set_user_video(self, video_path: str | None) -> None:
        """设置用户视频路径，用于与参考进行对比。"""
        self._state.user_video_path = video_path

    def _start_reference_extraction(self, video_path: str) -> None:
        """启动参考抽取后台线程，并连接进度/完成/失败信号。"""
        old_thread = self._ref_thread
        if _is_valid_thread(old_thread):
            old_thread.quit()
            old_thread.wait(1000)

        self._view.set_reference_ready(False)
        self._view.set_status("正在抽取标准动作（后台）…", 3000)

        thread = QThread()
        worker = ReferenceWorker(video_path, sample_fps=10.0)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_ref_progress)
        worker.finished.connect(self._on_ref_ready)
        worker.failed.connect(self._on_ref_failed)

        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)

        # 线程结束时，先清空引用，避免 close() 再去操作已 deleteLater 的对象
        thread.finished.connect(self._on_ref_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._ref_thread = thread
        self._ref_worker = worker

        thread.start()

    @Slot()
    def _on_ref_thread_finished(self) -> None:
        """清理后台线程引用。输入/输出：无。作用：避免悬挂引用。"""
        self._ref_thread = None
        self._ref_worker = None

    @Slot(int, int)
    def _on_ref_progress(self, cur: int, total: int) -> None:
        """更新状态栏进度显示。

        输入: cur 当前帧数，total 总帧数。
        输出: 无。
        作用: 向视图报告参考抽取进度。
        """
        if total > 0:
            self._view.set_status(f"标准动作抽取中：{cur}/{total}")

    @Slot(object)
    def _on_ref_ready(self, seq: object) -> None:
        """接收并保存参考姿态序列。

        输入: seq 期望为 list[PoseFrame]。
        输出: 无。
        作用: 校验类型并通知视图准备完成。
        """
        if not isinstance(seq, list):
            self._view.show_error("标准动作抽取失败", "抽取结果类型异常")
            return
        self._state.reference_sequence = seq
        self._view.set_reference_ready(True)
        self._view.set_status(f"标准动作准备完成（样本帧数：{len(seq)}）", 5000)

    @Slot(str)
    def _on_ref_failed(self, msg: str) -> None:
        """通知视图参考抽取失败原因。输入: 错误信息字符串。输出: 无。"""
        self._view.set_reference_ready(False)
        self._view.show_error("标准动作抽取失败", msg)

    def set_ref_preview_paused(self, paused: bool) -> None:
        """设置参考预览暂停状态。输入: paused。输出: 无。"""
        self._ref_preview_paused = paused

    def set_ref_preview_speed(self, speed: float) -> None:
        """设置参考预览播放倍速。输入: speed。输出: 无。"""
        if speed <= 0:
            return
        self._ref_preview_speed = speed

    def set_ref_preview_position_ms(self, pos_ms: int) -> None:
        """设置参考预览播放位置（毫秒）。输入: pos_ms。输出: 无。"""
        if pos_ms < 0:
            pos_ms = 0
        if self._ref_preview_duration_s > 0:
            max_ms = int(self._ref_preview_duration_s * 1000)
            pos_ms = min(pos_ms, max_ms)
        self._ref_preview_time_s = pos_ms / 1000.0
        self._ref_preview_last_ts = time.perf_counter()
        if self._cap_ref_preview is not None and self._cap_ref_preview.isOpened():
            self._cap_ref_preview.set(cv2.CAP_PROP_POS_MSEC, self._ref_preview_time_s * 1000.0)

    def is_running(self) -> bool:
        """是否正在练习。输入/输出: 无。作用: 供视图切换按钮状态。"""
        return self._state.running

    def stop_and_finalize(self, reason: str | None = None) -> None:
        """停止练习并计算平均相似度。输入: 可选原因。输出: 无。"""
        self.stop()
        avg = None
        if self._score_count > 0:
            avg = self._score_sum / self._score_count
            self._view.set_score(avg)
        if reason:
            self._view.set_status(reason, 3000)
        if avg is not None:
            self._view.set_status(f"本次平均相似度：{avg:.1f}", 5000)
        else:
            self._view.set_status("本次没有有效评分数据", 4000)
        self._view.set_running_state(False)

    def start(self) -> bool:
        """开始练习流程：打开用户输入、启动计时器与评分。

        输入/输出: 无。
        作用: 按需打开用户视频或摄像头，并开始周期性处理帧进行对比与显示。
        """
        if self._state.ref_video_path is None:
            self._view.show_error("缺少参考视频", "请先加载标准舞蹈视频")
            return False

        if self._state.reference_sequence is None:
            self._view.set_status("标准动作仍在准备中，先开始也可以，但分数会延后出现", 4000)

        self.stop()

        # 用户输入：优先用户视频，否则摄像头
        if self._state.user_video_path:
            self._cap_user = cv2.VideoCapture(self._state.user_video_path)
            self._user_is_file = True
        else:
            self._cap_user = cv2.VideoCapture(0)
            self._user_is_file = False

        if self._cap_user is None or not self._cap_user.isOpened():
            self._view.show_error("打开失败", "无法打开用户输入（摄像头/视频）")
            return False

        self._state.running = True
        self._state.start_time_s = time.perf_counter()
        self._score_ema = None
        self._score_sum = 0.0
        self._score_count = 0
        self._timer.start()
        self._view.set_status("开始练习：正在对比动作…", 2000)
        self._view.set_running_state(True)
        return True

    def stop(self) -> None:
        """停止练习：停计时器、释放用户输入。输入/输出：无。作用：清理运行状态。"""
        self._state.running = False
        if self._timer.isActive():
            self._timer.stop()

        if self._cap_user is not None:
            self._cap_user.release()
            self._cap_user = None

        # 不关闭参考预览cap（切换参考时会释放）

    def _on_tick(self) -> None:
        """主循环处理：同步时间、取参考、检测用户、评分、更新预览。

        输入/输出: 无。
        作用: 每帧按时间对齐参考，绘制骨架，计算分数并更新视图。
        """
        if not self._state.running:
            return

        # 0) 获取用户画面（优先读帧，保证 t_s 与“当前帧”一致）
        if self._cap_user is None:
            return
        ok, user_frame = self._cap_user.read()
        if not ok:
            self.stop()
            self._view.set_status("用户输入结束/读取失败，已停止", 5000)
            return

        # 1) 时间基准：文件用视频时间（读帧后再取 POS_MSEC 更准），摄像头用 perf_counter
        if self._user_is_file:
            t_s = float(self._cap_user.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        else:
            t_s = time.perf_counter() - self._state.start_time_s

        # 2) 取参考姿态（用于评分和参考画面叠加）
        ref_lm = None
        ref_t_s = t_s
        if self._ref_preview_duration_s > 0:
            ref_t_s = self._ref_preview_time_s
        if self._state.reference_sequence is not None:
            ref_lm = find_reference_by_time(self._state.reference_sequence, ref_t_s)

        # 3) 刷新参考视频预览（按独立播放进度，可暂停/倍速）
        if self._cap_ref_preview is not None and self._cap_ref_preview.isOpened():
            now = time.perf_counter()
            dt = max(0.0, now - self._ref_preview_last_ts)
            self._ref_preview_last_ts = now
            if not self._ref_preview_paused:
                self._ref_preview_time_s += dt * self._ref_preview_speed
                if self._ref_preview_duration_s > 0:
                    self._ref_preview_time_s = min(
                        self._ref_preview_time_s, self._ref_preview_duration_s
                    )

            self._cap_ref_preview.set(cv2.CAP_PROP_POS_MSEC, self._ref_preview_time_s * 1000.0)
            ok2, f = self._cap_ref_preview.read()
            if ok2:
                f2 = _draw_skeleton_bgr(f, ref_lm)
                self._view.set_ref_pixmap(_bgr_to_qpixmap(f2, 560, 420))

            if self._ref_preview_duration_s > 0:
                cur_ms = int(self._ref_preview_time_s * 1000)
                total_ms = int(self._ref_preview_duration_s * 1000)
                self._view.set_reference_progress(cur_ms, total_ms)

            if (
                self._ref_preview_duration_s > 0
                and self._ref_preview_time_s >= self._ref_preview_duration_s
                and not self._ref_preview_paused
            ):
                self.stop_and_finalize("标准视频播放结束，已停止")
                return

        # 4) 姿态检测 + 评分
        user_lm = self._detector.detect_landmarks(user_frame)
        score, _ = compare_angles(user_lm, ref_lm)
        if self._state.reference_sequence is not None:
            self._score_sum += score
            self._score_count += 1
            if self._score_ema is None:
                self._score_ema = score
            else:
                a = self._score_alpha
                self._score_ema = (1.0 - a) * self._score_ema + a * score
            self._view.set_score(self._score_ema)

        user_frame2 = _draw_skeleton_bgr(user_frame, user_lm)
        self._view.set_user_pixmap(_bgr_to_qpixmap(user_frame2, 560, 420))

    def close(self) -> None:
        """关闭控制器：停止流程、结束后台线程、释放资源。

        输入/输出: 无。
        作用: 应在窗口关闭时调用，确保无资源泄漏。
        """
        # 1) Stop timers + user capture
        self.stop()

        # 2) Stop background extraction thread
        thread = self._ref_thread
        if _is_valid_thread(thread):
            thread.quit()
            thread.wait(1500)
        self._ref_thread = None
        self._ref_worker = None

        # 3) Release reference preview capture
        if self._cap_ref_preview is not None:
            self._cap_ref_preview.release()
            self._cap_ref_preview = None

        # 4) Close detector
        self._detector.close()
