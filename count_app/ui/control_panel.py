"""
控制面板组件
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
                               QCheckBox, QPushButton, QSlider, QGridLayout)
from PySide6.QtCore import Qt, Signal


class ControlPanel(QWidget):
    """控制面板"""

    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 视频设置组
        video_group = self.create_video_settings()
        layout.addWidget(video_group)

        # 检测设置组
        detection_group = self.create_detection_settings()
        layout.addWidget(detection_group)

        # 计数设置组
        counting_group = self.create_counting_settings()
        layout.addWidget(counting_group)

        # 显示设置组
        display_group = self.create_display_settings()
        layout.addWidget(display_group)

        layout.addStretch()

    def create_video_settings(self) -> QGroupBox:
        """创建视频设置组"""
        group = QGroupBox("视频设置")
        layout = QGridLayout(group)

        # 分辨率设置
        layout.addWidget(QLabel("分辨率:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480", "1280x720", "1920x1080", "自动"
        ])
        self.resolution_combo.setCurrentText("1280x720")
        layout.addWidget(self.resolution_combo, 0, 1)

        # FPS设置
        layout.addWidget(QLabel("目标FPS:"), 1, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        layout.addWidget(self.fps_spin, 1, 1)

        # 摄像头选择
        layout.addWidget(QLabel("摄像头:"), 2, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2"])
        layout.addWidget(self.camera_combo, 2, 1)

        return group

    def create_detection_settings(self) -> QGroupBox:
        """创建检测设置组"""
        group = QGroupBox("姿态检测")
        layout = QGridLayout(group)

        # 置信度阈值
        layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.5)
        layout.addWidget(self.confidence_spin, 0, 1)

        # 可见度阈值
        layout.addWidget(QLabel("可见度阈值:"), 1, 0)
        self.visibility_spin = QDoubleSpinBox()
        self.visibility_spin.setRange(0.1, 1.0)
        self.visibility_spin.setSingleStep(0.1)
        self.visibility_spin.setValue(0.5)
        layout.addWidget(self.visibility_spin, 1, 1)

        # 平滑参数
        layout.addWidget(QLabel("平滑系数:"), 2, 0)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 1.0)
        self.smooth_spin.setSingleStep(0.1)
        self.smooth_spin.setValue(0.3)
        layout.addWidget(self.smooth_spin, 2, 1)

        return group

    def create_counting_settings(self) -> QGroupBox:
        """创建计数设置组"""
        group = QGroupBox("动作计数")
        layout = QGridLayout(group)

        # 角度容差
        layout.addWidget(QLabel("角度容差: "), 0, 0)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(5.0, 45.0)
        self.tolerance_spin.setSingleStep(5.0)
        self.tolerance_spin.setValue(15.0)
        self.tolerance_spin.setSuffix("°")
        layout.addWidget(self.tolerance_spin, 0, 1)

        # 最小间隔
        layout.addWidget(QLabel("最小间隔:"), 1, 0)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 5.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(0.5)
        self.interval_spin.setSuffix("s")
        layout.addWidget(self.interval_spin, 1, 1)

        # 目标动作数
        layout.addWidget(QLabel("目标动作数:"), 2, 0)
        self.target_count_spin = QSpinBox()
        self.target_count_spin.setRange(0, 1000)
        self.target_count_spin.setValue(0)
        self.target_count_spin.setSpecialValueText("自动检测")
        layout.addWidget(self.target_count_spin, 2, 1)

        return group

    def create_display_settings(self) -> QGroupBox:
        """创建显示设置组"""
        group = QGroupBox("显示设置")
        layout = QVBoxLayout(group)

        # 复选框选项
        self.show_skeleton_cb = QCheckBox("显示骨架")
        self.show_skeleton_cb.setChecked(True)
        layout.addWidget(self.show_skeleton_cb)

        self.show_angles_cb = QCheckBox("显示角度")
        self.show_angles_cb.setChecked(True)
        layout.addWidget(self.show_angles_cb)

        self.show_count_cb = QCheckBox("显示计数")
        self.show_count_cb.setChecked(True)
        layout.addWidget(self.show_count_cb)

        self.show_confidence_cb = QCheckBox("显示置信度")
        layout.addWidget(self.show_confidence_cb)

        # 重置按钮
        reset_btn = QPushButton("重置设置")
        reset_btn.clicked.connect(self.reset_settings)
        layout.addWidget(reset_btn)

        return group

    def connect_signals(self):
        """连接信号"""
        # 所有设置控件的变化都发送信号
        self.resolution_combo.currentTextChanged.connect(self.emit_settings)
        self.fps_spin.valueChanged.connect(self.emit_settings)
        self.camera_combo.currentTextChanged.connect(self.emit_settings)
        self.confidence_spin.valueChanged.connect(self.emit_settings)
        self.visibility_spin.valueChanged.connect(self.emit_settings)
        self.smooth_spin.valueChanged.connect(self.emit_settings)
        self.tolerance_spin.valueChanged.connect(self.emit_settings)
        self.interval_spin.valueChanged.connect(self.emit_settings)
        self.target_count_spin.valueChanged.connect(self.emit_settings)

        self.show_skeleton_cb.toggled.connect(self.emit_settings)
        self.show_angles_cb.toggled.connect(self.emit_settings)
        self.show_count_cb.toggled.connect(self.emit_settings)
        self.show_confidence_cb.toggled.connect(self.emit_settings)

    def emit_settings(self):
        """发送设置变化信号"""
        settings = self.get_all_settings()
        self.settings_changed.emit(settings)

    def get_all_settings(self) -> dict:
        """获取所有设置"""
        return {
            'resolution': self.resolution_combo.currentText(),
            'fps': self.fps_spin.value(),
            'camera_index': int(self.camera_combo.currentText()),
            'confidence_threshold': self.confidence_spin.value(),
            'visibility_threshold': self.visibility_spin.value(),
            'smooth_factor': self.smooth_spin.value(),
            'angle_tolerance': self.tolerance_spin.value(),
            'min_interval': self.interval_spin.value(),
            'target_count': self.target_count_spin.value(),
            'show_skeleton': self.show_skeleton_cb.isChecked(),
            'show_angles': self.show_angles_cb.isChecked(),
            'show_count': self.show_count_cb.isChecked(),
            'show_confidence': self.show_confidence_cb.isChecked()
        }

    def get_camera_index(self) -> int:
        """获取摄像头索引"""
        return int(self.camera_combo.currentText())

    def reset_settings(self):
        """重置所有设置"""
        self.resolution_combo.setCurrentText("1280x720")
        self.fps_spin.setValue(30)
        self.camera_combo.setCurrentText("0")
        self.confidence_spin.setValue(0.5)
        self.visibility_spin.setValue(0.5)
        self.smooth_spin.setValue(0.3)
        self.tolerance_spin.setValue(15.0)
        self.interval_spin.setValue(0.5)
        self.target_count_spin.setValue(0)

        self.show_skeleton_cb.setChecked(True)
        self.show_angles_cb.setChecked(True)
        self.show_count_cb.setChecked(True)
        self.show_confidence_cb.setChecked(False)