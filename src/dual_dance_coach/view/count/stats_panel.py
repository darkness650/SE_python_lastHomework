"""统计面板组件"""

import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StatsPanel(QWidget):
    """统计信息面板"""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.current_count = 0
        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        """设置UI"""
        layout = QHBoxLayout(self)

        # 左侧：动作统计
        count_group = self.create_count_display()
        layout.addWidget(count_group)

        # 右侧：时间统计
        time_group = self.create_time_stats()
        layout.addWidget(time_group)

    def create_count_display(self) -> QWidget:
        """创建计数显示组"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)

        # 大号LCD数字显示
        self.count_lcd = QLCDNumber(4)
        self.count_lcd.setMinimumHeight(60)
        self.count_lcd.display(0)
        self.count_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Filled)
        layout.addWidget(self.count_lcd)

        # 重置按钮
        reset_btn = QPushButton("重置计数")
        reset_btn.clicked.connect(self.reset_stats)
        layout.addWidget(reset_btn)

        return group

    def create_time_stats(self) -> QWidget:
        """创建时间统计组"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)

        runtime_layout = QHBoxLayout()
        runtime_layout.addWidget(QLabel("运行时间:"))
        self.runtime_label = QLabel("00:00:00")
        self.runtime_label.setFont(QFont("Courier", 12))
        runtime_layout.addWidget(self.runtime_label)
        layout.addLayout(runtime_layout)

        avg_rate_layout = QHBoxLayout()
        avg_rate_layout.addWidget(QLabel("平均速率:"))
        self.avg_rate_label = QLabel("0.0 次/分")
        avg_rate_layout.addWidget(self.avg_rate_label)
        layout.addLayout(avg_rate_layout)

        completion_layout = QHBoxLayout()
        completion_layout.addWidget(QLabel("完成率:"))
        self.completion_label = QLabel("处理中...")
        completion_layout.addWidget(self.completion_label)
        layout.addLayout(completion_layout)

        return group

    def setup_timer(self):
        """设置定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_runtime)
        self.update_timer.start(1000)  # 每秒更新

    def update_count(self, count: int, target: int = 0):
        """更新计数信息"""
        # 更新LCD显示
        self.count_lcd.display(count)
        self.current_count = count

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
            if elapsed > 0:
                avg_rate = (self.current_count / elapsed) * 60  # 次/分钟
                self.avg_rate_label.setText(f"{avg_rate:.1f} 次/分")

    def reset_stats(self):
        """重置统计，清零计数"""
        self.count_lcd.display(0)
        self.current_count = 0
        self.avg_rate_label.setText("0.0 次/分")
        self.completion_label.setText("处理中...")
        self.start_time = None
        self.runtime_label.setText("00:00:00")
