"""主窗口界面"""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dual_dance_coach.controller.count_controller import CountController
from dual_dance_coach.view.count.stats_panel import StatsPanel
from dual_dance_coach.view.count.video_widget import VideoDisplayWidget
from dual_dance_coach.view.resources import resolve_blob_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = CountController()
        self.current_process = None
        self._eval_running = False
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("动作计数分析器")
        self.resize(1920, 1080)

        # 创建中央部件
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")

        # 应用背景图（找不到文件时静默回退）
        bg_path = resolve_blob_path("countBG.png")
        if bg_path.exists() and bg_path.is_file():
            central_widget.setStyleSheet(
                central_widget.styleSheet()
                + "QWidget { background-color: transparent; } "
                + f"QWidget#centralWidget {{ background-image: url({bg_path.as_posix()}); background-repeat: no-repeat; background-position: center; }}"
            )
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 创建工作流程区域
        workflow_widget = self.create_workflow_area()
        main_layout.addWidget(workflow_widget)

        # 创建底部控制面板
        bottom_widget = self.create_bottom_area()
        main_layout.addWidget(bottom_widget)

        # 设置比例
        main_layout.setStretch(0, 5)  # 工作流程区域
        main_layout.setStretch(1, 1)  # 底部控制面板

        self.add_log("准备就绪 - 请先上传标准视频")

    def create_workflow_area(self) -> QWidget:
        """创建工作流程区域"""
        widget = QWidget()

        layout = QHBoxLayout(widget)

        # 步骤1：标准视频处理
        step1_widget = self.create_step1_widget()
        layout.addWidget(step1_widget)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # 步骤2：评测
        step2_widget = self.create_step2_widget()
        layout.addWidget(step2_widget)

        # 设置比例
        layout.setStretch(0, 1)
        layout.setStretch(2, 1)

        return widget

    def create_step1_widget(self) -> QWidget:
        """创建步骤1：标准视频处理"""
        widget = QGroupBox("步骤1：标准视频并生成动作模板(可多选)")
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)

        # 视频显示区域
        self.ref_video_display = VideoDisplayWidget()
        self.ref_video_display.setMinimumHeight(400)
        layout.addWidget(self.ref_video_display)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.btn_load_ref = QPushButton("选择标准视频")
        self.btn_load_ref.setMinimumHeight(40)

        self.btn_start_ref = QPushButton("开始处理")
        self.btn_start_ref.setMinimumHeight(40)
        self.btn_start_ref.setEnabled(False)

        self.btn_clear_ref = QPushButton("清空")
        self.btn_clear_ref.setMinimumHeight(40)

        button_layout.addWidget(self.btn_load_ref)
        button_layout.addWidget(self.btn_start_ref)
        button_layout.addWidget(self.btn_clear_ref)

        layout.addLayout(button_layout)

        # 目标次数输入
        targets_layout = QHBoxLayout()
        targets_layout.addWidget(QLabel("每个动作目标次数 (逗号分隔):"))
        self.targets_edit = QLineEdit()
        self.targets_edit.setPlaceholderText("例如: 10,8,12")
        targets_layout.addWidget(self.targets_edit)
        # 新增：确认按钮
        self.btn_confirm_targets = QPushButton("确认")
        self.btn_confirm_targets.setMinimumHeight(30)
        targets_layout.addWidget(self.btn_confirm_targets)
        layout.addLayout(targets_layout)

        # 处理信息显示
        self.ref_info_text = QTextEdit()
        self.ref_info_text.setMaximumHeight(80)
        self.ref_info_text.setReadOnly(True)
        self.ref_info_text.setPlaceholderText("模板信息将在这里显示...")
        layout.addWidget(self.ref_info_text)

        return widget

    def create_step2_widget(self) -> QWidget:
        """创建步骤2：评测"""
        widget = QGroupBox("步骤2：进行动作评测(达到目标自动切换)")
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)

        # 视频显示区域
        self.eval_video_display = VideoDisplayWidget()
        self.eval_video_display.setMinimumHeight(400)
        layout.addWidget(self.eval_video_display)

        # 评测控制面板
        eval_control_layout = self.create_eval_controls()
        layout.addLayout(eval_control_layout)

        # 评测按钮
        button_layout = QHBoxLayout()

        self.btn_load_eval = QPushButton("选择评测视频")
        self.btn_load_eval.setMinimumHeight(40)

        self.btn_start_eval = QPushButton("开始评测")
        self.btn_start_eval.setMinimumHeight(40)
        self.btn_start_eval.setEnabled(False)

        button_layout.addWidget(self.btn_load_eval)
        button_layout.addWidget(self.btn_start_eval)

        layout.addLayout(button_layout)

        # 评测结果显示
        self.eval_info_text = QTextEdit()
        self.eval_info_text.setMaximumHeight(80)
        self.eval_info_text.setReadOnly(True)
        self.eval_info_text.setPlaceholderText("评测信息将在这里显示...")
        layout.addWidget(self.eval_info_text)

        return widget

    def create_eval_controls(self) -> QVBoxLayout:
        """创建评测控制面板"""
        layout = QVBoxLayout()

        # 参数设置
        params_group = QGroupBox("评测参数")
        params_layout = QGridLayout(params_group)

        # 容差设置
        params_layout.addWidget(QLabel("角度容差: "), 0, 0)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(5.0, 45.0)
        self.tolerance_spin.setValue(10.0)
        self.tolerance_spin.setSuffix("°")
        params_layout.addWidget(self.tolerance_spin, 0, 1)

        # 关键动作数
        params_layout.addWidget(QLabel("关键动作数:"), 0, 2)
        self.key_actions_spin = QSpinBox()
        self.key_actions_spin.setRange(0, 100)
        self.key_actions_spin.setValue(0)
        self.key_actions_spin.setSpecialValueText("自动检测")
        params_layout.addWidget(self.key_actions_spin, 0, 3)

        # 评测来源选择
        params_layout.addWidget(QLabel("评测来源:"), 1, 0)
        source_layout = QHBoxLayout()
        self.eval_source_video_rb = QRadioButton("评测视频")
        self.eval_source_camera_rb = QRadioButton("摄像头")
        self.eval_source_video_rb.setChecked(True)
        source_layout.addWidget(self.eval_source_video_rb)
        source_layout.addWidget(self.eval_source_camera_rb)
        params_layout.addLayout(source_layout, 1, 1, 1, 3)

        layout.addWidget(params_group)

        return layout

    def create_bottom_area(self) -> QWidget:
        """创建底部区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(6)

        # 统计面板
        self.stats_panel = StatsPanel()
        layout.addWidget(self.stats_panel)

        # 日志区域
        log_group = QWidget()
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(120)
        self.log_display.setReadOnly(True)
        self.log_display.append("运行日志")
        log_layout.addWidget(self.log_display)

        layout.addWidget(log_group)

        # 设置比例
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)

        return widget

    def connect_signals(self):
        """连接信号槽"""
        # 步骤1信号
        self.btn_load_ref.clicked.connect(self.load_reference_video)
        self.btn_start_ref.clicked.connect(self.start_reference_processing)
        self.btn_clear_ref.clicked.connect(self.clear_reference)
        # 新增：确认目标次数
        self.btn_confirm_targets.clicked.connect(self.confirm_targets_binding)

        # 步骤2信号
        self.btn_load_eval.clicked.connect(self.load_evaluation_video)
        self.btn_start_eval.clicked.connect(self.start_evaluation)
        self.eval_source_video_rb.toggled.connect(self.on_eval_source_changed)
        self.eval_source_camera_rb.toggled.connect(self.on_eval_source_changed)

    def load_reference_video(self):
        """加载标准视频（支持多选）"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择标准视频（可多选）", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if files:
            self.reference_video_paths = files
            self.btn_start_ref.setEnabled(True)
            self.add_log(f"已选择 {len(files)} 个标准视频")
            self.add_log("下一步：设置目标次数并开始处理")

    def start_reference_processing(self):
        """开始处理标准视频（列表）"""
        if not hasattr(self, "reference_video_paths") or not self.reference_video_paths:
            QMessageBox.warning(self, "错误", "请先选择至少一个标准视频")
            return

        # 解析目标次数（可选）
        targets = self.parse_targets_input()
        if targets:
            # 先设置到控制器（长度不匹配将被截断）
            self.controller.set_action_targets(targets)

        # 开始处理
        self.btn_start_ref.setEnabled(False)
        self.add_log("开始处理标准视频...")

        # 创建处理线程
        self.ref_process_thread = ReferenceProcessThread(
            self.controller, self.reference_video_paths
        )
        self.ref_process_thread.frame_ready.connect(self.ref_video_display.update_frame)
        self.ref_process_thread.info_ready.connect(self.update_reference_info)
        self.ref_process_thread.finished.connect(self.on_reference_finished)
        self.ref_process_thread.start()

    def parse_targets_input(self):
        """解析逗号分隔的目标次数输入为整数列表"""
        text = self.targets_edit.text().strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",") if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(max(0, int(p)))
            except Exception:
                pass
        return vals

    def confirm_targets_binding(self):
        """确认并绑定每个动作目标次数到控制器，提供与已选标准视频数量的校验提示。"""
        targets = self.parse_targets_input()
        if not targets:
            QMessageBox.information(self, "提示", "请填写目标次数，例如：10,8,12")
            return
        # 绑定到控制器（控制器内部会在模板构建后按模板数量对齐）
        self.controller.set_action_targets(targets)
        # 校验数量并提示
        selected_n = len(getattr(self, "reference_video_paths", []))
        if selected_n == 0:
            self.add_log("已绑定目标次数，但尚未选择标准视频")
            return
        if len(targets) != selected_n:
            self.add_log(
                f"目标次数数量({len(targets)})与标准视频数({selected_n})不一致，已按较少的一侧对齐"
            )
        else:
            self.add_log("目标次数已确认并与每个动作绑定")

    def clear_reference(self):
        """清空标准视频"""
        if hasattr(self, "reference_video_paths"):
            del self.reference_video_paths
        self.ref_video_display.clear()
        self.ref_info_text.clear()
        self.targets_edit.clear()
        self.btn_start_ref.setEnabled(False)
        self.add_log("已清空标准视频")

    def load_evaluation_video(self):
        """加载评测视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择评测视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if file_path:
            self.evaluation_video_path = file_path
            self.eval_source_video_rb.setChecked(True)
            self.update_eval_button_state()
            self.add_log(f"已选择评测视频: {file_path}")

    def on_eval_source_changed(self):
        """评测来源切换"""
        self.update_eval_button_state()

    def update_eval_button_state(self):
        """更新评测按钮状态"""
        if self._eval_running:
            self.btn_start_eval.setEnabled(True)
            return

        has_template = (self.controller.ref_template is not None) or bool(
            self.controller.ref_templates
        )
        if self.eval_source_camera_rb.isChecked():
            has_source = True
        else:
            has_source = hasattr(self, "evaluation_video_path")
        self.btn_start_eval.setEnabled(has_template and has_source)

    def start_evaluation(self):
        """开始评测"""
        if getattr(self, "_eval_running", False):
            self.stop_evaluation()
            return

        if (self.controller.ref_template is None) and (not self.controller.ref_templates):
            QMessageBox.warning(self, "错误", "请先处理标准视频生成模板")
            return

        # 获取参数
        tolerance = self.tolerance_spin.value()
        key_actions = self.key_actions_spin.value()
        if key_actions == 0:
            key_actions = None

        # 若用户在此时输入/修改了目标次数，再次同步
        targets = self.parse_targets_input()
        if targets:
            self.controller.set_action_targets(targets)

        use_webcam = self.eval_source_camera_rb.isChecked()
        eval_file: str | None = None if use_webcam else getattr(self, "evaluation_video_path", None)
        if (not use_webcam) and (not eval_file):
            QMessageBox.warning(self, "错误", "请选择评测视频或切换到摄像头")
            return

        # 开始评测
        self._eval_running = True
        self.btn_start_eval.setText("停止评测")
        self.stats_panel.start_timing()

        source_info = "摄像头" if use_webcam else "视频文件"
        self.add_log(f"开始评测 - 使用{source_info}")

        # 创建评测线程
        self.eval_process_thread = EvaluationProcessThread(
            self.controller, eval_file, use_webcam, tolerance, key_actions
        )
        self.eval_process_thread.frame_ready.connect(self.eval_video_display.update_frame)
        self.eval_process_thread.info_ready.connect(self.update_evaluation_info)
        self.eval_process_thread.count_updated.connect(self.stats_panel.update_count)
        self.eval_process_thread.finished.connect(self.on_evaluation_finished)
        self.eval_process_thread.start()

    def stop_evaluation(self):
        """停止评测"""
        if hasattr(self, "eval_process_thread"):
            self.controller.stop()
            self.eval_process_thread.wait()
        self.on_evaluation_finished()
        self.add_log("评测已停止")

    def update_reference_info(self, info):
        """更新标准视频信息"""
        self.ref_info_text.append(info)

    def update_evaluation_info(self, info):
        """更新评测信息"""
        self.eval_info_text.append(info)

    def on_reference_finished(self):
        """标准视频处理完成"""
        self.btn_start_ref.setEnabled(True)
        self.add_log("标准视频处理完成，模板已生成")
        self.add_log("下一步：选择评测来源并开始评测")
        self.update_eval_button_state()

    def on_evaluation_finished(self):
        """评测完成"""
        self._eval_running = False
        self.btn_start_eval.setText("开始评测")
        self.update_eval_button_state()
        self.stats_panel.stop_timing()
        if self.eval_source_video_rb.isChecked():
            self.add_log("视频评测完成")

    def add_log(self, message: str):
        """添加日志"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """关闭事件"""
        if hasattr(self, "ref_process_thread"):
            self.ref_process_thread.wait()
        if hasattr(self, "eval_process_thread"):
            self.controller.stop()
            self.eval_process_thread.wait()
        from dual_dance_coach.view.main_window import show_home_window

        show_home_window()
        event.accept()


class ReferenceProcessThread(QThread):
    """标准视频处理线程，支持多视频文件"""

    frame_ready = Signal(object, object, dict)  # 修改：使用Signal
    info_ready = Signal(str)  # 修改：使用Signal

    def __init__(self, controller: CountController, video_paths: list[str]):
        super().__init__()
        self.controller = controller
        self.video_paths = video_paths  # 列表

    def run(self):
        """运行处理"""
        try:
            for info, frame in self.controller.start_reference(self.video_paths):
                self.info_ready.emit(info)
                if frame is not None:
                    self.frame_ready.emit(frame, None, {})
        except Exception as e:
            self.info_ready.emit(f"处理失败: {e}")


class EvaluationProcessThread(QThread):
    """评测处理线程，支持视频文件和摄像头"""

    frame_ready = Signal(object, object, dict)  # 修改：使用Signal
    info_ready = Signal(str)  # 修改：使用Signal
    count_updated = Signal(int, int)  # 修改：使用Signal

    def __init__(
        self,
        controller: CountController,
        eval_file: str | None,
        use_webcam: bool,
        tolerance: float,
        key_actions: int | None,
    ):
        super().__init__()
        self.controller = controller
        self.eval_file = eval_file
        self.use_webcam = use_webcam
        self.tolerance = tolerance
        self.key_actions = key_actions

    def run(self):
        """运行评测"""
        try:
            count = 0
            for info, frame in self.controller.start_template_evaluation(
                self.eval_file, self.use_webcam, self.tolerance, self.key_actions
            ):
                self.info_ready.emit(info)
                if frame is None:
                    continue

                # 动作 2/2 | 目标:10 | 模板进度: 0/60 | 已完成: 10 | 匹配: ✓ | 前瞻跳过:0 | 处理FPS:8.7
                # 从info中提取计数信息
                if "已完成:" in info:
                    try:
                        count_str = info.split("已完成: ")[1].split("|")[0].strip()
                        new_count = int(count_str)
                        if "目标" in info:
                            target_str = info.split("目标:")[1].split("|")[0].strip()
                            target = int(target_str)
                            if new_count != count:
                                count = new_count
                                self.count_updated.emit(count, target)
                    except Exception as e:
                        print(e)
                self.frame_ready.emit(frame, None, {"count": count})
        except Exception as e:
            self.info_ready.emit(f"评测失败: {e}")
