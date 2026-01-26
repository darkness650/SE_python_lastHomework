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

## 项目结构（视图/业务分离）

这是暂时的项目结构，基本完成后可能还会调整。

```
├───src
│   └───dual_dance_coach
│       ├───core    # 核心纯业务逻辑
│       └───view    # 界面与控制器
│           ├───count       # 健身计数相关
│           │   ├───mvc     # 有关的业务逻辑
│           │   └───ui      # Qt界面
│           └───dance       # 舞蹈练习相关
│               ├───controller      # ui的控制器
│               └───ui              # Qt界面
└───test    # 测试代码
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
pyinstaller --noconfirm --clean --name dual_dance_coach --windowed --add-data "blob/pose_landmarker_heavy.task;blob" src/dual_dance_coach/__main__.py
```

从环境的`Lib/site-packages`目录下复制 MediaPipe 的动态链接库文件。

```bash
cp <path-to-your-python-env>/Lib/site-packages/mediapipe dist/dual_dance_coach/_internal/mediapipe -r
```

即可运行`dist/dual_dance_coach/dual_dance_coach.exe`。
