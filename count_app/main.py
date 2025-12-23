"""
动作计数客户端 - 主入口
基于原有Gradio MVC架构
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QDir

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("动作计数分析器")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Dance Analysis")

    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(__file__), ".temp")
    os.makedirs(temp_dir, exist_ok=True)

    # 创建主窗口
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())