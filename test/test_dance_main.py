import os
import sys

from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from dual_dance_coach.view.dance.main_window import MainWindow


def main() -> int:
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="light_blue.xml")
    app.setStyleSheet(app.styleSheet() + "\n* { font-size: 16pt; }")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
