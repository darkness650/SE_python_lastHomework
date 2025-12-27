import sys
import cv2
import time
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox
)

class VideoCaptureThread(QThread):
    frame_data = Signal(QImage)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_data.emit(qt_image)
            time.sleep(0.03)  # 约30帧每秒
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class CameraRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头录制 (PySide6)")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton("打开摄像头")
        self.record_button = QPushButton("开始录制")
        self.save_button = QPushButton("保存视频")
        self.record_button.setEnabled(False)
        self.save_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.save_button)

        self.capture_thread = None
        self.recording = False
        self.recorded_frames = []
        self.fps = 30
        self.frame_size = (640, 480)

        self.start_button.clicked.connect(self.start_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.save_button.clicked.connect(self.save_video)

    def start_camera(self):
        if self.capture_thread and self.capture_thread.isRunning():
            return
        self.capture_thread = VideoCaptureThread()
        self.capture_thread.frame_data.connect(self.update_image)
        self.capture_thread.start()
        self.start_button.setEnabled(False)
        self.record_button.setEnabled(True)

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.recording:
            w = qt_image.width()
            h = qt_image.height()
            arr = np.array(qt_image.bits(), dtype=np.uint8).reshape((h, w, 3))
            bgr_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.recorded_frames.append(bgr_arr)
            self.frame_size = (w, h)

    def toggle_recording(self):
        if not self.recording:
            self.recorded_frames = []
            self.recording = True
            self.record_button.setText("停止录制")
            self.save_button.setEnabled(False)
        else:
            self.recording = False
            self.record_button.setText("开始录制")
            self.save_button.setEnabled(True)

    def save_video(self):
        if not self.recorded_frames:
            QMessageBox.warning(self, "警告", "没有录制内容！")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存视频", "", "视频文件 (*.avi)")
        if filename:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, self.fps, self.frame_size)
            for frame in self.recorded_frames:
                out.write(frame)
            out.release()
            QMessageBox.information(self, "提示", f"视频已保存：{filename}")
            self.save_button.setEnabled(False)

    def closeEvent(self, event):
        if self.capture_thread:
            self.capture_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraRecorder()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())