from __future__ import annotations

"""MediaPipe Pose 的常用骨架连接（仅用于画面叠加显示）。

索引遵循 MediaPipe Pose 33 关键点定义。
这里不依赖 mediapipe 包本身，便于业务/界面层解耦、减少类型检查噪音。
"""
# 连接对 (a, b)
POSE_CONNECTIONS: set[tuple[int, int]] = {
    # torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # left arm
    (11, 13),
    (13, 15),
    # right arm
    (12, 14),
    (14, 16),
    # left leg
    (23, 25),
    (25, 27),
    # right leg
    (24, 26),
    (26, 28),
    # small extras (feet)
    (27, 31),
    (31, 29),
    (28, 32),
    (32, 30),
}
