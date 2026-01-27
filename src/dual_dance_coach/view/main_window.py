import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

_home_window_ref = None


BACKGROUND_IMAGE_PATH = "./blob/background.png"


def show_home_window() -> "MainWindow":
    global _home_window_ref
    if _home_window_ref is None:
        _home_window_ref = MainWindow()
    _home_window_ref.show()
    _home_window_ref.raise_()
    _home_window_ref.activateWindow()
    return _home_window_ref


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        global _home_window_ref
        self.setWindowTitle("主界面")
        self.resize(1920, 1080)

        self._root = QWidget(self)
        self._root.setObjectName("homeRoot")
        self.setCentralWidget(self._root)

        self.has_background = self._apply_background(BACKGROUND_IMAGE_PATH)
        self._build_ui()

        if _home_window_ref is None:
            _home_window_ref = self

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self._root)
        main_layout.setContentsMargins(48, 48, 48, 48)
        main_layout.setSpacing(24)

        # ===== 上半：Logo/标题 =====
        top_box = QWidget(self._root)
        top_box.setObjectName("topBox")
        top_layout = QVBoxLayout(top_box)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if not self.has_background:
            # 无背景图时显示标题文字
            self.lbl_title = QLabel("舞动乾坤")
            self.lbl_title.setObjectName("lblTitle")
            self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.lbl_subtitle = QLabel("舞蹈辅助练习与健身计数")
            self.lbl_subtitle.setObjectName("lblSubtitle")
            self.lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

            top_layout.addWidget(self.lbl_title)
            top_layout.addWidget(self.lbl_subtitle)

        # ===== 下半：两个大按钮 =====
        bottom_box = QWidget(self._root)
        bottom_box.setObjectName("bottomBox")
        bottom_layout = QHBoxLayout(bottom_box)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(36)
        bottom_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_dance = QPushButton("舞蹈练习")
        self.btn_dance.setObjectName("btnDance")
        self.btn_dance.setMinimumSize(420, 220)
        self.btn_dance.clicked.connect(self._open_dance_window)

        self.btn_count = QPushButton("健身计数")
        self.btn_count.setObjectName("btnCount")
        self.btn_count.setMinimumSize(420, 220)
        self.btn_count.clicked.connect(self._open_count_window)

        bottom_layout.addWidget(self.btn_dance)
        bottom_layout.addWidget(self.btn_count)

        main_layout.addWidget(top_box, stretch=1)
        main_layout.addWidget(bottom_box, stretch=1)

        # ===== 样式（qt-material 主题下用 stylesheet） =====
        self._root.setStyleSheet(
            self._root.styleSheet() + "QWidget#homeRoot { background-color: transparent; }"
            "QWidget#topBox, QWidget#bottomBox { background-color: transparent; }"
            "QLabel#lblTitle { font-size: 64pt; font-weight: 800; }"
            "QLabel#lblSubtitle { font-size: 28pt; font-weight: 500; }"
            "QPushButton#btnDance, QPushButton#btnCount { font-size: 30pt; font-weight: 700; }"
        )

    def _apply_background(self, image_path: str | None) -> bool:
        """设置背景图占位（后续可在此处填入图片路径）"""
        if image_path and os.path.exists(image_path) and os.path.isfile(image_path):
            self._root.setStyleSheet(
                self._root.styleSheet()
                + f"QWidget#homeRoot {{ background-image: url('{image_path}'); "
                "background-position: center; background-repeat: no-repeat; }}"
            )
            return True
        return False

    def _open_dance_window(self) -> None:
        from dual_dance_coach.view.dance.main_window import MainWindow as DanceMainWindow

        self._dance_window = DanceMainWindow()
        self._dance_window.show()
        self.hide()

    def _open_count_window(self) -> None:
        from dual_dance_coach.view.count.main_window import MainWindow as CountMainWindow

        self._count_window = CountMainWindow()
        self._count_window.show()
        self.hide()

    def closeEvent(self, event) -> None:
        event.accept()
