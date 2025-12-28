"""
主窗口界面 - 基于原有MVC模式 (修复PySide6信号问题，扩展多标准视频与目标次数)
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QTabWidget, QLabel, QPushButton, QProgressBar,
                               QTextEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
                               QGridLayout, QSlider, QComboBox, QCheckBox,
                               QFileDialog, QMessageBox, QSplitter, QFrame, QLineEdit)
from PySide6.QtCore import Qt, QTimer, QThread, Signal  # 修改：使用Signal而不是pyqtSignal
from PySide6.QtGui import QPixmap, QFont, QIcon

from .video_widget import VideoDisplayWidget
from .control_panel import ControlPanel
from .stats_panel import StatsPanel
from  ..mvc.controller import CountController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = CountController()
        self.current_process = None
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("动作计数分析器 - 基于MVC架构")
        self.setGeometry(100, 100, 1600, 1000)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建工作流程区域
        workflow_widget = self.create_workflow_area()
        main_layout.addWidget(workflow_widget)

        # 创建底部控制面板
        bottom_widget = self.create_bottom_area()
        main_layout.addWidget(bottom_widget)

        # 设置比例
        main_layout.setStretch(0, 4)  # 工作流程区域
        main_layout.setStretch(1, 1)  # 底部控制面板

        # 创建状态栏
        self.statusBar().showMessage("准备就绪 - 请先上传标准视频")

    def create_workflow_area(self) -> QWidget:
        """创建工作流程区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # 步骤1：标准视频处理
        step1_widget = self.create_step1_widget()
        layout.addWidget(step1_widget)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
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
        widget = QGroupBox("步骤1：标准视频处理")
        layout = QVBoxLayout(widget)

        # 标题说明
        title_label = QLabel("上传标准视频并生成动作模板（可多选）")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        # 视频显示区域
        self.ref_video_display = VideoDisplayWidget()
        self.ref_video_display.setMinimumHeight(400)
        layout.addWidget(self.ref_video_display)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.btn_load_ref = QPushButton("选择标准视频（多选）")
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
        self.ref_info_text.setMaximumHeight(120)
        self.ref_info_text.setReadOnly(True)
        self.ref_info_text.setPlaceholderText("模板信息将在这里显示...")
        layout.addWidget(QLabel("模板信息:"))
        layout.addWidget(self.ref_info_text)

        return widget

    def create_step2_widget(self) -> QWidget:
        """创建步骤2：评测"""
        widget = QGroupBox("步骤2：动作评测")
        layout = QVBoxLayout(widget)

        # 标题说明
        title_label = QLabel("基于生成的模板进行动作评测（达到目标自动切换）")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

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

        self.btn_start_camera = QPushButton("启用摄像头")
        self.btn_start_camera.setMinimumHeight(40)

        self.btn_start_eval = QPushButton("开始评测")
        self.btn_start_eval.setMinimumHeight(40)
        self.btn_start_eval.setEnabled(False)

        self.btn_stop_eval = QPushButton("停止评测")
        self.btn_stop_eval.setMinimumHeight(40)
        self.btn_stop_eval.setEnabled(False)

        button_layout.addWidget(self.btn_load_eval)
        button_layout.addWidget(self.btn_start_camera)
        button_layout.addWidget(self.btn_start_eval)
        button_layout.addWidget(self.btn_stop_eval)

        layout.addLayout(button_layout)

        # 评测结果显示
        self.eval_info_text = QTextEdit()
        self.eval_info_text.setMaximumHeight(120)
        self.eval_info_text.setReadOnly(True)
        self.eval_info_text.setPlaceholderText("评测信息将在这里显示...")
        layout.addWidget(QLabel("评测信息:"))
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

        # 摄像头选择
        params_layout.addWidget(QLabel("摄像头:"), 1, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2"])
        params_layout.addWidget(self.camera_combo, 1, 1)

        # 使用摄像头复选框
        self.use_webcam_cb = QCheckBox("使用摄像头实时评测")
        params_layout.addWidget(self.use_webcam_cb, 1, 2, 1, 2)

        layout.addWidget(params_group)

        return layout

    def create_bottom_area(self) -> QWidget:
        """创建底部区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # 统计面板
        self.stats_panel = StatsPanel()
        layout.addWidget(self.stats_panel)

        # 日志区域
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)

        # 清空日志按钮
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_btn)

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
        self.btn_start_camera.clicked.connect(self.setup_camera)
        self.btn_start_eval.clicked.connect(self.start_evaluation)
        self.btn_stop_eval.clicked.connect(self.stop_evaluation)

        # 摄像头复选框
        self.use_webcam_cb.toggled.connect(self.on_webcam_toggle)

    def load_reference_video(self):
        """加载标准视频（支持多选）"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择标准视频（可多选）", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if files:
            self.reference_video_paths = files
            self.btn_start_ref.setEnabled(True)
            self.add_log(f"已选择 {len(files)} 个标准视频")
            self.statusBar().showMessage("标准视频已选择，请设置目标次数并开始处理")

    def start_reference_processing(self):
        """开始处理标准视频（列表）"""
        if not hasattr(self, 'reference_video_paths') or not self.reference_video_paths:
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
        self.ref_process_thread.frame_ready.connect(
            self.ref_video_display.update_frame
        )
        self.ref_process_thread.info_ready.connect(
            self.update_reference_info
        )
        self.ref_process_thread.finished.connect(
            self.on_reference_finished
        )
        self.ref_process_thread.start()

    def parse_targets_input(self):
        """解析逗号分隔的目标次数输入为整数列表"""
        text = self.targets_edit.text().strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(',') if p.strip()]
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
        selected_n = len(getattr(self, 'reference_video_paths', []))
        if selected_n == 0:
            self.add_log("已绑定目标次数，但尚未选择标准视频")
            self.statusBar().showMessage("目标次数已确认（请先选择标准视频）")
            return
        if len(targets) != selected_n:
            self.add_log(f"目标次数数量({len(targets)})与标准视频数({selected_n})不一致，已按较少的一侧对齐")
            self.statusBar().showMessage("目标次数已确认（数量不一致将截断或忽略多余）")
        else:
            self.add_log("目标次数已确认并与每个动作绑定")
            self.statusBar().showMessage("目标次数已确认")

    def clear_reference(self):
        """清空标准视频"""
        if hasattr(self, 'reference_video_paths'):
            del self.reference_video_paths
        self.ref_video_display.clear()
        self.ref_info_text.clear()
        self.targets_edit.clear()
        self.btn_start_ref.setEnabled(False)
        self.add_log("已清空标准视频")
        self.statusBar().showMessage("请选择标准视频")

    def load_evaluation_video(self):
        """加载评测视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择评测视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if file_path:
            self.evaluation_video_path = file_path
            self.use_webcam_cb.setChecked(False)
            self.update_eval_button_state()
            self.add_log(f"已选择评测视频: {file_path}")

    def setup_camera(self):
        """设置摄像头"""
        camera_index = int(self.camera_combo.currentText())
        self.camera_index = camera_index
        self.use_webcam_cb.setChecked(True)
        if hasattr(self, 'evaluation_video_path'):
            del self.evaluation_video_path
        self.update_eval_button_state()
        self.add_log(f"已选择摄像头 {camera_index}")

    def on_webcam_toggle(self, checked):
        """摄像头复选框切换"""
        if checked:
            self.btn_load_eval.setEnabled(False)
            if not hasattr(self, 'camera_index'):
                self.camera_index = 0
        else:
            self.btn_load_eval.setEnabled(True)
            if hasattr(self, 'camera_index'):
                del self.camera_index
        self.update_eval_button_state()

    def update_eval_button_state(self):
        """更新评测按钮状态"""
        has_template = (self.controller.ref_template is not None) or bool(self.controller.ref_templates)
        has_source = (hasattr(self, 'evaluation_video_path') or
                      hasattr(self, 'camera_index'))
        self.btn_start_eval.setEnabled(has_template and has_source)

    def start_evaluation(self):
        """开始评测"""
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

        use_webcam = self.use_webcam_cb.isChecked()
        eval_file = None if use_webcam else getattr(self, 'evaluation_video_path', None)

        # 开始评测
        self.btn_start_eval.setEnabled(False)
        self.btn_stop_eval.setEnabled(True)
        self.stats_panel.start_timing()

        source_info = "摄像头" if use_webcam else "视频文件"
        self.add_log(f"开始评测 - 使用{source_info}")

        # 创建评测线程
        self.eval_process_thread = EvaluationProcessThread(
            self.controller, eval_file, use_webcam, tolerance, key_actions
        )
        self.eval_process_thread.frame_ready.connect(
            self.eval_video_display.update_frame
        )
        self.eval_process_thread.info_ready.connect(
            self.update_evaluation_info
        )
        self.eval_process_thread.count_updated.connect(
            self.stats_panel.update_count
        )
        self.eval_process_thread.finished.connect(
            self.on_evaluation_finished
        )
        self.eval_process_thread.start()

    def stop_evaluation(self):
        """停止评测"""
        if hasattr(self, 'eval_process_thread'):
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
        self.statusBar().showMessage("模板已就绪，可以开始评测")
        self.update_eval_button_state()

    def on_evaluation_finished(self):
        """评测完成"""
        self.btn_start_eval.setEnabled(True)
        self.btn_stop_eval.setEnabled(False)
        self.stats_panel.stop_timing()
        if not self.use_webcam_cb.isChecked():
            self.add_log("视频评测完成")

    def add_log(self, message: str):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")

    def clear_log(self):
        """清空日志"""
        self.log_display.clear()

    def closeEvent(self, event):
        """关闭事件"""
        if hasattr(self, 'ref_process_thread'):
            self.ref_process_thread.wait()
        if hasattr(self, 'eval_process_thread'):
            self.controller.stop()
            self.eval_process_thread.wait()
        event.accept()


class ReferenceProcessThread(QThread):
    """标准视频处理线程"""
    frame_ready = Signal(object, object, dict)  # 修改：使用Signal
    info_ready = Signal(str)  # 修改：使用Signal

    def __init__(self, controller, video_paths):
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
    """评测处理线程"""
    frame_ready = Signal(object, object, dict)  # 修改：使用Signal
    info_ready = Signal(str)  # 修改：使用Signal
    count_updated = Signal(int, int, float)  # 修改：使用Signal

    def __init__(self, controller, eval_file, use_webcam, tolerance, key_actions):
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
                if frame is not None:
                    # 从info中提取计数信息
                    if "已完成:" in info:
                        try:
                            count_str = info.split("已完成: ")[1].split("|")[0].strip()
                            new_count = int(count_str)
                            if new_count != count:
                                count = new_count
                                self.count_updated.emit(count, 0, 0.0)
                        except Exception as e:
                            print(e)
                    self.frame_ready.emit(frame, None, {'count': count})
        except Exception as e:
            self.info_ready.emit(f"评测失败: {e}")