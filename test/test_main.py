import faulthandler
import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from dual_dance_coach.view.main_window import show_home_window


def main_test() -> int:
    # PyInstaller --windowed 会导致 sys.stderr/sys.stdout 为 None，导致日志初始化报错
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    log_path = Path.cwd() / "app-crash.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )

    faulthandler.enable()

    def _excepthook(exc_type, exc, tb):
        logging.exception("未捕获异常", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook

    def _qt_message_handler(mode, context, message):
        logging.info("[Qt] %s", message)

    qInstallMessageHandler(_qt_message_handler)

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="light_blue.xml")
    app.setStyleSheet(app.styleSheet() + "\n* { font-size: 12pt; }")

    app.aboutToQuit.connect(lambda: logging.info("应用退出"))

    # 创建主窗口
    show_home_window()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main_test())
