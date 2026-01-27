# 舞动乾坤 —— 舞蹈辅助练习与健身计数

MediaPipe + PySide6 + OpenCV

基于 MediaPipe Pose 做肢体关键点/关节角度对比：

- 舞蹈部分
  - 选择“标准舞蹈视频”作为参考动作
  - 选择用户输入（摄像头或用户视频）
  - 实时输出动作相似度分数，并在画面上叠加骨架
  - 随时暂停/继续，调整进度条
  - 随时结束跟舞，并在结束时输出最终得分

- 健身计数
  - 选择标准动作视频作为参考动作（该视频中只能包含一次该动作）
  - 选择用户输入（摄像头或视频）
  - 实时输出目前做了多少个参考动作

## 项目结构

```
SE_python_lastHomework
│   .gitignore
│   pyproject.toml      # Python 项目配置文件
│   README.md       # 项目说明文档
│   TODO.md
│
├───blob            # 资源文件夹
│       background.png      # 主界面背景图
│       pose_landmarker_heavy.task  # MediaPipe Pose 模型文件
│
├───src     # 源代码文件夹
│   └───dual_dance_coach    # 主程序包
│       │   __init__.py
│       │   __main__.py     # 程序入口
│       │
│       ├───controller      # 控制器模块
│       │       count_controller.py     # 动作计数控制器
│       │       practice_controller.py  # 舞蹈练习控制器
│       │       view_protocol.py        # 舞蹈练习视图协议
│       │       __init__.py
│       │
│       ├───core        # 核心逻辑模块
│       │       motion_counter.py       # 动作计数核心逻辑
│       │       pose_connections.py     # MediaPipe Pose 连接定义
│       │       pose_detector.py        # MediaPipe Pose 检测封装
│       │       reference_extractor.py  # 参考动作提取器
│       │       scoring.py          # 舞蹈评分算法
│       │       types.py            # 核心数据类型定义
│       │       __init__.py
│       │
│       └───view        # 视图模块
│           │   main_window.py      # 主界面
│           │   __init__.py
│           │
│           ├───count       # 动作计数视图模块
│           │       main_window.py      # 动作计数主界面
│           │       stats_panel.py      # 统计面板
│           │       video_widget.py     # 视频播放组件
│           │       __init__.py
│           │
│           └───dance       # 舞蹈练习视图模块
│                   main_window.py      # 舞蹈练习主界面
│                   __init__.py
│
└───test
        remove_pycache.py
        test_count_main.py
        test_cwd.py
        test_dance_main.py
        __init__.py
```

## 开发和测试

### 环境

- Python: >=3.12
- 依赖见 `pyproject.toml`

### 安装

在项目目录执行：

```bash
pip install -e .
```

### 测试

舞蹈动作识别

```bash
python ./test/test_dance_main.py
```

健身动作计数

```bash
python ./test/test_count_main.py
```

### 打包

```bash
pyinstaller --noconfirm --clean --name dual_dance_coach --windowed --add-data "blob/pose_landmarker_heavy.task;blob" --add-data "blob/background.png;blob" src/dual_dance_coach/__main__.py
```

从环境的`Lib/site-packages`目录下复制 MediaPipe 的动态链接库文件：

```bash
cp <path-to-your-python-env>/Lib/site-packages/mediapipe dist/dual_dance_coach/_internal/mediapipe -r
```

即可运行`dist/dual_dance_coach/dual_dance_coach.exe`。
