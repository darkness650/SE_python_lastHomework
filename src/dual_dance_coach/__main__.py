import os
import sys

from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from dual_dance_coach.view.main_window import show_home_window


def main() -> int:
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="light_blue.xml")
    app.setStyleSheet(app.styleSheet() + "\n* { font-size: 12pt; }")

    # 创建主窗口
    show_home_window()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
