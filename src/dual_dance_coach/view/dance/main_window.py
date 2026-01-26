from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from dual_dance_coach.controller.practice_controller import PracticeController


@dataclass
class UiState:
    ref_video_path: str | None = None
    user_video_path: str | None = None
    ref_ready: bool = False


class MainWindow(QMainWindow):
    def __init__(self):
        """初始化主窗口并构造控制器。

        输入/输出: 无。
        作用: 设置窗口属性，创建控制器并搭建 UI 与事件绑定。
        """
        super().__init__()
        self.setWindowTitle("舞蹈辅助练习")
        self.resize(1920, 1080)

        self._state = UiState()
        self._controller = PracticeController(self)
        self._ref_slider_dragging = False

        self._build_ui()
        self._wire_events()

    def _build_ui(self) -> None:
        """构建界面控件与布局。

        输入/输出: 无。
        作用: 初始化按钮、标签、布局与状态栏。
        """
        root = QWidget(self)
        self.setCentralWidget(root)

        # ===== 顶部控制栏 =====
        top_bar = QGroupBox("控制栏")
        top_bar.setMinimumHeight(192)
        top_layout = QHBoxLayout(top_bar)

        # 左侧：标准视频控制
        left_box = QWidget(top_bar)
        left_layout = QHBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.btn_load_ref = QPushButton("选择标准视频")
        self.ref_progress = QSlider(Qt.Orientation.Horizontal)
        self.ref_progress.setMinimum(0)
        self.ref_progress.setMaximum(1000)
        self.ref_progress.setValue(0)
        self.ref_progress.setTracking(True)
        self.ref_progress.setMinimumWidth(240)
        self.btn_ref_pause = QPushButton("暂停")
        self.cmb_speed = QComboBox()
        self.cmb_speed.setEditable(False)
        self.cmb_speed.setMinimumWidth(120)
        for v in [0.25, 0.5, 1.0, 1.5, 2.0]:
            self.cmb_speed.addItem(f"{v:.2f}", v)
        self.cmb_speed.setCurrentText("1.00")
        lbl_speed = QLabel("倍速")
        lbl_speed.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        left_layout.addWidget(self.btn_load_ref)
        left_layout.addWidget(self.ref_progress, stretch=1)
        left_layout.addWidget(self.btn_ref_pause)
        left_layout.addWidget(lbl_speed)
        left_layout.addWidget(self.cmb_speed)

        # 中间：开始按钮
        center_box = QWidget(top_bar)
        center_layout = QHBoxLayout(center_box)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_start = QPushButton("开始跳舞")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setStyleSheet("QPushButton#btnStart { font-size: 28pt; font-weight: 700; }")
        self.btn_start.setMinimumHeight(100)
        self.btn_start.setMinimumWidth(200)
        self.btn_start.setEnabled(False)
        center_layout.addWidget(self.btn_start)

        # 右侧：用户视频控制
        right_box = QWidget(top_bar)
        right_layout = QHBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.radio_cam = QRadioButton("摄像头模式")
        self.radio_user = QRadioButton("用户视频模式")
        self.radio_cam.setChecked(True)
        self.btn_load_user = QPushButton("选择用户视频")

        self.user_mode_group = QButtonGroup(self)
        self.user_mode_group.addButton(self.radio_cam)
        self.user_mode_group.addButton(self.radio_user)

        right_layout.addStretch(1)
        right_layout.addWidget(self.radio_cam)
        right_layout.addWidget(self.radio_user)
        right_layout.addWidget(self.btn_load_user)

        top_layout.addWidget(left_box, stretch=1)
        top_layout.addWidget(center_box, stretch=1)
        top_layout.addWidget(right_box, stretch=1)

        # ===== 中间预览区 =====
        middle_box = QWidget(root)
        middle_layout = QHBoxLayout(middle_box)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(24)

        self.lbl_ref = QLabel("标准视频预览")
        self.lbl_ref.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref.setMinimumSize(800, 520)

        self.lbl_user = QLabel("用户画面预览")
        self.lbl_user.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_user.setMinimumSize(800, 520)

        middle_layout.addWidget(self.lbl_ref, stretch=1)
        middle_layout.addWidget(self.lbl_user, stretch=1)

        # ===== 底部信息区 =====
        bottom_bar = QGroupBox("信息")
        bottom_bar.setMinimumHeight(192)
        bottom_layout = QHBoxLayout(bottom_bar)

        self.lbl_next_step = QLabel("下一步：请选择标准视频")
        self.lbl_next_step.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_next_step.setWordWrap(True)

        score_box = QWidget(bottom_bar)
        score_layout = QVBoxLayout(score_box)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_score_title = QLabel("相似度")
        self.lbl_score_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score_title.setObjectName("lblScoreTitle")

        self.lbl_score = QLabel("--")
        self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score.setObjectName("lblScore")
        self.lbl_score.setStyleSheet(
            "QLabel#lblScore { font-size: 36pt; font-weight: 700; color: #ff6a00; }"
        )

        score_layout.addWidget(self.lbl_score_title)
        score_layout.addWidget(self.lbl_score)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(200)

        bottom_layout.addWidget(self.lbl_next_step, stretch=1)
        bottom_layout.addWidget(score_box, stretch=1)
        bottom_layout.addWidget(self.log_panel, stretch=1)

        layout = QVBoxLayout(root)
        layout.addWidget(top_bar)
        layout.addWidget(middle_box, stretch=1)
        layout.addWidget(bottom_bar)

    def _wire_events(self) -> None:
        """将控件事件连接到相应的槽函数。

        输入/输出: 无。
        作用: 绑定按钮点击事件到业务逻辑。
        """
        self.btn_load_ref.clicked.connect(self._on_load_ref)
        self.btn_load_user.clicked.connect(self._on_load_user)
        self.btn_start.clicked.connect(self._on_start_cam)
        self.btn_ref_pause.clicked.connect(self._on_toggle_ref_pause)
        self.cmb_speed.currentIndexChanged.connect(self._on_ref_speed_changed)
        self.ref_progress.sliderPressed.connect(self._on_ref_slider_pressed)
        self.ref_progress.sliderReleased.connect(self._on_ref_slider_released)
        self.ref_progress.sliderMoved.connect(self._on_ref_slider_moved)
        self.radio_cam.toggled.connect(self._on_user_mode_changed)
        self.radio_user.toggled.connect(self._on_user_mode_changed)

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
        self._state.ref_ready = False
        self.set_status(f"已选择标准视频：{path}", 5000)
        self._controller.load_reference(path)
        self.btn_ref_pause.setText("暂停")
        self._controller.set_ref_preview_paused(False)
        self._controller.set_ref_preview_speed(1.0)
        self.cmb_speed.setCurrentText("1.00")
        self.ref_progress.setValue(0)
        self._set_next_step("标准视频已选择，正在抽取动作…")
        self._update_start_enabled()

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
        self.set_status(f"已选择用户视频：{path}", 5000)
        self._controller.set_user_video(path)
        self.radio_user.setChecked(True)
        self._set_next_step("用户视频已选择，可以开始跳舞")
        self._update_start_enabled()

    def _on_start_cam(self) -> None:
        """开始练习流程（摄像头/用户视频）。

        输入/输出: 无。
        作用: 若已选择参考视频，则委托控制器启动练习。
        """
        if self._controller.is_running():
            self._on_stop()
            return
        if not self._state.ref_video_path:
            QMessageBox.information(self, "提示", "请先加载标准舞蹈视频")
            return
        if not self._state.ref_ready:
            QMessageBox.information(self, "提示", "标准视频尚未处理完成，请稍候")
            return
        if self.radio_user.isChecked() and not self._state.user_video_path:
            QMessageBox.information(self, "提示", "请先选择用户视频，或切换到摄像头模式")
            return
        if self._controller.start():
            self._set_next_step("正在检测…请保持在画面中")

    def _on_stop(self) -> None:
        """停止练习流程。输入/输出: 无。作用: 委托控制器停止。"""
        self._controller.stop_and_finalize("已停止")

    def _on_toggle_ref_pause(self) -> None:
        """切换参考预览暂停/继续（仅UI状态）。"""
        if self.btn_ref_pause.text() == "暂停":
            self.btn_ref_pause.setText("继续")
            self._controller.set_ref_preview_paused(True)
        else:
            self.btn_ref_pause.setText("暂停")
            self._controller.set_ref_preview_paused(False)

    def _on_ref_speed_changed(self) -> None:
        """参考预览倍速变更。"""
        speed = self.cmb_speed.currentData()
        if isinstance(speed, (int, float)):
            self._controller.set_ref_preview_speed(float(speed))

    def _on_ref_slider_pressed(self) -> None:
        """拖动参考进度条开始。"""
        self._ref_slider_dragging = True

    def _on_ref_slider_released(self) -> None:
        """拖动参考进度条结束。"""
        self._ref_slider_dragging = False
        self._controller.set_ref_preview_position_ms(self.ref_progress.value())

    def _on_ref_slider_moved(self, value: int) -> None:
        """拖动参考进度条中，实时更新预览位置。"""
        self._controller.set_ref_preview_position_ms(value)

    def _on_user_mode_changed(self) -> None:
        """用户模式切换：摄像头/用户视频。"""
        if self.radio_cam.isChecked():
            self._state.user_video_path = None
            self._controller.set_user_video(None)
            self._set_next_step("已选择摄像头模式，点击开始跳舞")
        elif self.radio_user.isChecked() and not self._state.user_video_path:
            self._set_next_step("请选择用户视频，或切换到摄像头模式")
        self._update_start_enabled()

    def _set_next_step(self, message: str) -> None:
        self.lbl_next_step.setText(f"下一步：{message}")

    def _update_start_enabled(self) -> None:
        ready = bool(self._state.ref_video_path) and self._state.ref_ready
        if self.radio_user.isChecked():
            ready = ready and bool(self._state.user_video_path)
        self.btn_start.setEnabled(ready)

    # ====== 供控制器调用（视图接口） ======

    def set_score(self, score_0_100: float) -> None:
        """显示当前相似度分数。

        输入: 0~100 分。
        输出: 无。
        作用: 更新顶部标签文本。
        """
        self.lbl_score.setText(f"{score_0_100:.1f}")

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
        if message:
            self.log_panel.appendPlainText(message)

    def set_reference_ready(self, ready: bool) -> None:
        """更新标准视频就绪状态。输入: ready。输出: 无。"""
        self._state.ref_ready = ready
        if ready:
            self._set_next_step("标准动作准备完成，可以开始跳舞")
        else:
            self._set_next_step("标准动作正在准备中…")
        self._update_start_enabled()

    def set_reference_progress(self, cur: int, total: int) -> None:
        """更新参考视频进度条。输入: cur/total。输出: 无。"""
        if getattr(self, "_ref_slider_dragging", False):
            return
        if total <= 0:
            self.ref_progress.setRange(0, 1000)
            return
        self.ref_progress.setRange(0, total)
        self.ref_progress.setValue(cur)

    def set_running_state(self, running: bool) -> None:
        """更新运行状态，切换开始按钮文本。"""
        if running:
            self.btn_start.setText("停止")
        else:
            self.btn_start.setText("开始跳舞")

    def closeEvent(self, event) -> None:
        """窗口关闭钩子：释放控制器资源后再关闭。

        输入: event。
        输出: 无。
        作用: 确保后台线程与设备释放。
        """
        try:
            if self._controller.is_running():
                self._on_stop()
            self._controller.close()
        finally:
            super().closeEvent(event)
            from dual_dance_coach.view.main_window import show_home_window

            show_home_window()
