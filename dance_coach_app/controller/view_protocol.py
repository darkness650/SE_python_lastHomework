from __future__ import annotations

from typing import Protocol

from PySide6.QtGui import QPixmap


class PracticeView(Protocol):
    """练习视图接口：控制器通过该协议调用视图更新。"""
    def set_score(self, score_0_100: float) -> None:
        """显示分数。输入: 0~100 浮点值。输出: 无。作用: 更新分数显示。"""
        ...

    def set_status(self, message: str, timeout_ms: int = 3000) -> None:
        """更新状态栏。输入: 文本与超时毫秒。输出: 无。作用: 提示进度/状态。"""
        ...

    def show_error(self, title: str, message: str) -> None:
        """显示错误弹窗。输入: 标题与内容。输出: 无。"""
        ...

    def set_ref_pixmap(self, pixmap: QPixmap) -> None:
        """更新参考预览图。输入: QPixmap。输出: 无。"""
        ...

    def set_user_pixmap(self, pixmap: QPixmap) -> None:
        """更新用户预览图。输入: QPixmap。输出: 无。"""
        ...
