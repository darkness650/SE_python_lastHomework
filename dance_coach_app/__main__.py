from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from dance_coach_app.ui.main_window import MainWindow


def main() -> int:
    """应用入口：创建 QApplication 与主窗口并运行事件循环。

    输入/输出: 无显式输入；返回应用退出码（int）。
    作用: 启动 GUI 程序并阻塞直到窗口关闭。
    """
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    # main()
    raise SystemExit(main())
