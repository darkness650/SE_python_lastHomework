from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,

    QGroupBox,
    QHBoxLayout,
    QLabel,

    QPushButton,

    QVBoxLayout,
    QWidget, QGridLayout, QStatusBar, QMessageBox, QMainWindow,
)

from dance_coach_app.controller.practice_controller import PracticeController


@dataclass
class UiState:
    ref_video_path: Optional[str] = None
    user_video_path: Optional[str] = None


class MainWindow(QMainWindow):
    def __init__(self):
        """初始化主窗口并构造控制器。

        输入/输出: 无。
        作用: 设置窗口属性，创建控制器并搭建 UI 与事件绑定。
        """
        super().__init__()
        self.setWindowTitle("舞蹈辅助练习（MediaPipe + PySide6）")
        self.resize(1200, 700)

        self._state = UiState()
        self._controller = PracticeController(self)

        self._build_ui()
        self._wire_events()

    def _build_ui(self) -> None:
        """构建界面控件与布局。

        输入/输出: 无。
        作用: 初始化按钮、标签、布局与状态栏。
        """
        root = QWidget(self)
        self.setCentralWidget(root)

        self.btn_load_ref = QPushButton("加载标准视频")
        self.btn_load_user = QPushButton("加载用户视频（可选）")
        self.btn_start_cam = QPushButton("开始摄像头")
        self.btn_stop = QPushButton("停止")

        self.lbl_ref = QLabel("参考视频预览")
        self.lbl_ref.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref.setMinimumSize(520, 360)

        self.lbl_user = QLabel("用户画面预览")
        self.lbl_user.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_user.setMinimumSize(520, 360)

        self.lbl_score = QLabel("相似度：--")
        self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        grp_controls = QGroupBox("操作")
        controls_layout = QHBoxLayout(grp_controls)
        controls_layout.addWidget(self.btn_load_ref)
        controls_layout.addWidget(self.btn_load_user)
        controls_layout.addWidget(self.btn_start_cam)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.lbl_score)

        grid = QGridLayout()
        grid.addWidget(self.lbl_ref, 0, 0)
        grid.addWidget(self.lbl_user, 0, 1)

        layout = QVBoxLayout(root)
        layout.addWidget(grp_controls)
        layout.addLayout(grid)

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)

    def _wire_events(self) -> None:
        """将控件事件连接到相应的槽函数。

        输入/输出: 无。
        作用: 绑定按钮点击事件到业务逻辑。
        """
        self.btn_load_ref.clicked.connect(self._on_load_ref)
        self.btn_load_user.clicked.connect(self._on_load_user)
        self.btn_start_cam.clicked.connect(self._on_start_cam)
        self.btn_stop.clicked.connect(self._on_stop)

    def _on_load_ref(self) -> None:
        """选择并加载参考视频文件。

        输入/输出: 无（通过文件对话框）。
        作用: 更新状态并通知控制器加载参考视频。
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标准舞蹈视频",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not path:
            return
        self._state.ref_video_path = path
        self._status.showMessage(f"已选择标准视频：{path}", 5000)
        self._controller.load_reference(path)

    def _on_load_user(self) -> None:
        """选择并设置用户视频文件。

        输入/输出: 无（通过文件对话框）。
        作用: 更新状态并通知控制器使用该用户视频作为输入。
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择用户跳舞视频（可选）",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not path:
            return
        self._state.user_video_path = path
        self._status.showMessage(f"已选择用户视频：{path}", 5000)
        self._controller.set_user_video(path)

    def _on_start_cam(self) -> None:
        """开始练习流程（摄像头/用户视频）。

        输入/输出: 无。
        作用: 若已选择参考视频，则委托控制器启动练习。
        """
        if not self._state.ref_video_path:
            QMessageBox.information(self, "提示", "请先加载标准舞蹈视频")
            return
        self._controller.start()

    def _on_stop(self) -> None:
        """停止练习流程。输入/输出: 无。作用: 委托控制器停止。"""
        self._controller.stop()

    # ====== 供控制器调用（视图接口） ======

    def set_score(self, score_0_100: float) -> None:
        """显示当前相似度分数。

        输入: 0~100 分。
        输出: 无。
        作用: 更新顶部标签文本。
        """
        self.lbl_score.setText(f"相似度：{score_0_100:.1f}")

    def set_ref_pixmap(self, pixmap: QPixmap) -> None:
        """更新参考视频预览图片。输入: QPixmap。输出: 无。"""
        self.lbl_ref.setPixmap(pixmap)

    def set_user_pixmap(self, pixmap: QPixmap) -> None:
        """更新用户视频/摄像头预览图片。输入: QPixmap。输出: 无。"""
        self.lbl_user.setPixmap(pixmap)

    def show_error(self, title: str, message: str) -> None:
        """弹出错误消息框。输入: 标题与内容。输出: 无。"""
        QMessageBox.critical(self, title, message)

    def set_status(self, message: str, timeout_ms: int = 3000) -> None:
        """更新状态栏消息。输入: 文本与超时毫秒。输出: 无。"""
        self.statusBar().showMessage(message, timeout_ms)

    def closeEvent(self, event) -> None:
        """窗口关闭钩子：释放控制器资源后再关闭。

        输入: event。
        输出: 无。
        作用: 确保后台线程与设备释放。
        """
        try:
            self._controller.close()
        finally:
            super().closeEvent(event)
