"""
视频显示组件 - 支持高清显示
"""
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QFont
import cv2
import numpy as np


class VideoDisplayWidget(QWidget):
    """高清视频显示组件"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                background-color: #f5f5f5;
                border-radius: 8px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("等待视频输入...")
        self.video_label.setScaledContents(False)  # 保持宽高比

        layout.addWidget(self.video_label)

    def update_frame(self, frame: np.ndarray, landmarks=None, annotations=None):
        """更新视频帧"""
        if frame is None:
            return

        # 转换为QPixmap并显示
        pixmap = self.numpy_to_qpixmap(frame)
        if pixmap:
            # 按比例缩放以适应显示区域
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

    def numpy_to_qpixmap(self, frame: np.ndarray) -> QPixmap:
        """将numpy数组转换为QPixmap"""
        try:
            if len(frame.shape) == 3:
                height, width, channels = frame.shape
                if channels == 3:
                    # BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    q_image = QImage(
                        rgb_frame.data, width, height,
                        width * channels, QImage.Format_RGB888
                    )
                else:
                    q_image = QImage(
                        frame.data, width, height,
                        width * channels, QImage.Format_ARGB32
                    )
            else:
                height, width = frame.shape
                q_image = QImage(
                    frame.data, width, height,
                    width, QImage.Format_Grayscale8
                )

            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"转换图像失败: {e}")
            return QPixmap()

    def clear(self):
        """清空显示"""
        self.video_label.clear()
        self.video_label.setText("等待视频输入...")