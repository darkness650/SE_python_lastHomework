"""
统计面板组件
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QLabel, QProgressBar, QLCDNumber, QGridLayout,
                               QPushButton)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import time


class StatsPanel(QWidget):
    """统计信息面板"""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 计数显示组
        count_group = self.create_count_display()
        layout.addWidget(count_group)

        # 时间统计组
        time_group = self.create_time_stats()
        layout.addWidget(time_group)

        layout.addStretch()

    def create_count_display(self) -> QGroupBox:
        """创建计数显示组"""
        group = QGroupBox("动作计数")
        layout = QVBoxLayout(group)

        # 大号LCD数字显示
        self.count_lcd = QLCDNumber(4)
        self.count_lcd.setMinimumHeight(80)
        self.count_lcd.display(0)
        self.count_lcd.setSegmentStyle(QLCDNumber.Filled)
        layout.addWidget(self.count_lcd)

        # 统计信息
        stats_layout = QGridLayout()

        stats_layout.addWidget(QLabel("当前计数:"), 0, 0)
        self.current_count_label = QLabel("0")
        self.current_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        stats_layout.addWidget(self.current_count_label, 0, 1)

        stats_layout.addWidget(QLabel("完成率:"), 1, 0)
        self.completion_label = QLabel("处理中...")
        stats_layout.addWidget(self.completion_label, 1, 1)

        layout.addLayout(stats_layout)

        # 重置按钮
        reset_btn = QPushButton("重置计数")
        reset_btn.clicked.connect(self.reset_stats)
        layout.addWidget(reset_btn)

        return group

    def create_time_stats(self) -> QGroupBox:
        """创建时间统计组"""
        group = QGroupBox("时间统计")
        layout = QGridLayout(group)

        # 运行时间
        layout.addWidget(QLabel("运行时间:"), 0, 0)
        self.runtime_label = QLabel("00:00:00")
        self.runtime_label.setFont(QFont("Courier", 12))
        layout.addWidget(self.runtime_label, 0, 1)

        # 平均速率
        layout.addWidget(QLabel("平均速率: "), 1, 0)
        self.avg_rate_label = QLabel("0.0 次/分")
        layout.addWidget(self.avg_rate_label, 1, 1)

        return group

    def setup_timer(self):
        """设置定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_runtime)
        self.update_timer.start(1000)  # 每秒更新

    def update_count(self, count: int, target: int = None, rate: float = None):
        """更新计数信息"""
        # 更新LCD显示
        self.count_lcd.display(count)
        self.current_count_label.setText(str(count))

        # 更新完成状态
        if target and target > 0:
            completion = (count / target) * 100
            self.completion_label.setText(f"{completion:.1f}%")
        else:
            self.completion_label.setText("实时计数中...")

    def start_timing(self):
        """开始计时"""
        self.start_time = time.time()

    def stop_timing(self):
        """停止计时"""
        self.start_time = None

    def update_runtime(self):
        """更新运行时间"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.runtime_label.setText(time_str)

            # 计算平均速率
            current_count = int(self.current_count_label.text())
            if elapsed > 0:
                avg_rate = (current_count / elapsed) * 60  # 次/分钟
                self.avg_rate_label.setText(f"{avg_rate:.1f} 次/分")

    def reset_stats(self):
        """重置统计"""
        self.count_lcd.display(0)
        self.current_count_label.setText("0")
        self.avg_rate_label.setText("0.0 次/分")
        self.completion_label.setText("处理中...")
        self.start_time = None
        self.runtime_label.setText("00:00:00")