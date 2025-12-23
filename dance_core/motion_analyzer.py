"""
动作分析核心模块
"""
from PySide6.QtCore import QObject, QThread, QTimer, Signal
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any
import os

# 导入你现有的模块（需要根据实际路径调整）
try:
    from dance_core.pose_detector import PoseDetector, PoseDetectorConfig
    from dance_core.motion_counter import AngleTemplate, TemplateRepetitionCounter, TemplateMatcherConfig
    from dance_core.types import PoseLandmarks
    from dance_core.scoring import compute_joint_angles
except ImportError:
    # 如果无法导入，创建占位符类
    print("警告: 无法导入dance_core模块，使用占位符")


    class PoseDetector:
        def __init__(self, config=None):
            pass

        def detect_landmarks(self, frame):
            return None

        def close(self):
            pass


    class PoseDetectorConfig:
        pass


    class AngleTemplate:
        @staticmethod
        def from_video(path, sample_fps=10, visibility_th=0.5):
            return None


    class TemplateRepetitionCounter:
        def __init__(self, template, config, key_actions=None):
            self.count = 0

        def process_frame(self, landmarks):
            return self.count


    class TemplateMatcherConfig:
        def __init__(self, tolerance_deg=10, visibility_th=0.5):
            pass


    class PoseLandmarks:
        def __init__(self, data):
            self.data = data


    def compute_joint_angles(landmarks, visibility_threshold=0.5):
        return {}


class MotionAnalyzer(QObject):
    """动作分析器"""

    # 信号定义
    frame_processed = Signal(np.ndarray, object, dict)  # frame, landmarks, annotations
    count_updated = Signal(int, int, float)  # current_count, target_count, rate
    status_changed = Signal(str)
    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.pose_detector = None
        self.motion_counter = None
        self.ref_template = None

        # 视频捕获
        self.video_capture = None
        self.is_camera_mode = False

        # 处理线程
        self.processing_thread = None
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_frame)

        # 设置
        self.settings = {
            'resolution': '1280x720',
            'fps': 30,
            'confidence_threshold': 0.5,
            'visibility_threshold': 0.5,
            'angle_tolerance': 15.0,
        'min_interval': 0.5,
        'target_count': 0,
        'show_skeleton': True,
        'show_angles': True,
        'show_count': True
        }

        # 统计信息
        self.frame_count = 0
        self.last_count_time = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        self.initialize_detector()

    def initialize_detector(self):
        """初始化姿态检测器"""
        try:
            config = PoseDetectorConfig()
            self.pose_detector = PoseDetector(config)
            self.log_message.emit("姿态检测器初始化成功")
        except Exception as e:
            self.log_message.emit(f"姿态检测器初始化失败: {e}")

    def update_settings(self, new_settings: Dict[str, Any]):
        """更新设置"""
        self.settings.update(new_settings)
        self.log_message.emit("设置已更新")

        # 如果正在运行，重新配置
        if self.video_capture and self.video_capture.isOpened():
            self.apply_video_settings()

    def apply_video_settings(self):
        """应用视频设置"""
        if not self.video_capture or not self.video_capture.isOpened():
            return

        # 设置分辨率
        resolution = self.settings['resolution']
        if resolution != '自动':
            try:
                w, h = map(int, resolution.split('x'))
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                self.log_message.emit(f"分辨率设置为:  {w}x{h}")
            except:
                self.log_message.emit("分辨率设置失败，使用默认值")

        # 设置FPS
        fps = self.settings['fps']
        if self.is_camera_mode:
            self.video_capture.set(cv2.CAP_PROP_FPS, fps)

        # 设置其他属性
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟

    def load_reference_video(self, file_path: str) -> bool:
        """加载参考视频"""
        try:
            # 构建参考模板
            self.ref_template = AngleTemplate.from_video(
                file_path,
                sample_fps=self.settings['fps'] / 2,  # 使用一半的FPS采样
                visibility_th=self.settings['visibility_threshold']
            )

            # 初始化动作计数器
            config = TemplateMatcherConfig(
                tolerance_deg=self.settings['angle_tolerance'],
                visibility_th=self.settings['visibility_threshold']
            )

            self.motion_counter = TemplateRepetitionCounter(
                self.ref_template,
                config,
                key_actions=self.settings['target_count'] if self.settings['target_count'] > 0 else None
            )

            self.log_message.emit(f"参考视频加载成功: {file_path}")
            self.status_changed.emit("参考模板已就绪")
            return True

        except Exception as e:
            self.log_message.emit(f"参考视频加载失败: {e}")
            return False

    def load_test_video(self, file_path: str) -> bool:
        """加载测试视频"""
        try:
            if self.video_capture:
                self.video_capture.release()

            self.video_capture = cv2.VideoCapture(file_path)
            self.is_camera_mode = False

            if not self.video_capture.isOpened():
                raise Exception("无法打开视频文件")

            # 获取视频信息
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            self.log_message.emit(f"测试视频加载成功: {file_path}")
            self.log_message.emit(f"视频信息: {fps:. 1f}FPS, {duration:.1f}秒")

            # 开始处理
            self.start_processing()
            return True

        except Exception as e:
            self.log_message.emit(f"测试视频加载失败: {e}")
            return False

    def start_camera(self, camera_index: int = 0) -> bool:
        """启动摄像头"""
        try:
            if self.video_capture:
                self.video_capture.release()

            self.video_capture = cv2.VideoCapture(camera_index)
            self.is_camera_mode = True

            if not self.video_capture.isOpened():
                raise Exception(f"无法打开摄像头 {camera_index}")

            # 应用设置
            self.apply_video_settings()

            # 获取实际分辨率
            w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)

            self.log_message.emit(f"摄像头启动成功:  {w}x{h} @ {actual_fps:. 1f}FPS")

            # 开始处理
            self.start_processing()
            return True

        except Exception as e:
            self.log_message.emit(f"摄像头启动失败: {e}")
            return False

    def start_processing(self):
        """开始处理"""
        if not self.video_capture or not self.video_capture.isOpened():
            self.log_message.emit("没有可用的视频源")
            return

        # 设置处理间隔
        interval = 1000 // self.settings['fps']  # 转换为毫秒
        self.processing_timer.start(interval)

        self.frame_count = 0
        self.last_fps_time = time.time()

        self.status_changed.emit("正在处理...")
        self.log_message.emit("开始处理视频")

    def process_frame(self):
        """处理单帧"""
        if not self.video_capture or not self.video_capture.isOpened():
            self.stop()
            return

        ret, frame = self.video_capture.read()
        if not ret:
            if not self.is_camera_mode:
                self.log_message.emit("视频播放完毕")
                self.stop()
            return

        try:
            # 姿态检测
            landmarks = None
            if self.pose_detector:
                lm_data = self.pose_detector.detect_landmarks(frame)
                if lm_data is not None:
                    landmarks = PoseLandmarks(lm_data)

            # 动作计数
            current_count = 0
            if self.motion_counter and landmarks:
                current_count = self.motion_counter.process_frame(landmarks)

            # 计算角度信息
            angles = {}
            if landmarks and landmarks.data is not None:
                angles = compute_joint_angles(
                    landmarks.data,
                    visibility_threshold=self.settings['visibility_threshold']
                )

            # 准备标注信息
            annotations = {
                'count': current_count,
                **{f"{k}_angle": v for k, v in angles.items()}
            }

            # 更新统计
            self.frame_count += 1
            current_time = time.time()

            # 计算FPS
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            # 计算速率
            rate = 0.0
            if current_time - self.last_count_time > 0:
                rate = current_count / ((current_time - self.last_count_time) / 60)

            # 发送信号
            self.frame_processed.emit(frame, landmarks, annotations)
            self.count_updated.emit(
                current_count,
                self.settings['target_count'],
                rate
            )

        except Exception as e:
            self.log_message.emit(f"处理帧时出错: {e}")

    def stop(self):
        """停止处理"""
        self.processing_timer.stop()

        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        self.status_changed.emit("已停止")
        self.log_message.emit("处理已停止")

    def __del__(self):
        """析构函数"""
        self.stop()
        if self.pose_detector:
            self.pose_detector.close()