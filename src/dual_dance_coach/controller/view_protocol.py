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

    def set_reference_ready(self, ready: bool) -> None:
        """更新参考视频就绪状态。输入: ready。输出: 无。作用: 控制开始按钮。"""
        ...

    def set_reference_progress(self, cur: int, total: int) -> None:
        """更新参考抽取进度。输入: cur/total。输出: 无。作用: 刷新进度条。"""
        ...

    def set_running_state(self, running: bool) -> None:
        """更新运行状态。输入: running。输出: 无。作用: 切换开始/停止按钮状态。"""
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
