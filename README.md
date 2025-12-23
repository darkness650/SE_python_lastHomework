# 舞蹈辅助练习与健身计数（MediaPipe + PySide6 + gradio）

基于 MediaPipe Pose 做肢体关键点/关节角度对比：

- 舞蹈部分
  
  - 选择“标准舞蹈视频”作为参考动作
  
  - 选择用户输入（摄像头或用户视频）
  
  - 实时输出动作相似度分数，并在画面上叠加骨架

- 健身计数
  
  - 选择标准动作视频作为参考动作（该视频中只能包含一次该动作）
  
  - 选择用户输入（摄像头或视频）
  
  - 实时输出目前做了多少个参考动作

## 环境

- Python: 3.12.11
- OS: Windows

## 安装

在项目目录执行：

如果你使用 conda（你当前环境看起来是 Miniforge/Conda）：

```powershell
conda activate <你的环境>
python -m pip install -U pip
pip install -r requirements.txt
python -c "import sys; print(sys.executable)"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

```

> 若 `mediapipe` 在你机器上暂时没有适配 3.12 的 wheel，可尝试：
> 
> - 升级到较新版本的 `mediapipe`
> - 或使用 conda-forge 的 mediapipe 包（如果你使用 conda）

## 运行

舞蹈动作识别

```powershell
python -m dance_coach_app
```

健身动作计数

```powershell
python -m count_app.main
```



## 最短使用流程

- 点击“加载标准视频”，选择老师/标准动作视频（建议镜头全身、光线充足）
- （可选）点击“加载用户视频”选择你自己的视频；不选则默认使用摄像头
- 点击“开始摄像头”（开始练习），右侧显示用户画面与相似度分数
- 点击“停止”结束

## 结构（视图/业务分离）

- `dance_core/`：纯业务逻辑（姿态检测、参考动作提取、评分）
- `dance_coach_app/`：跳舞计分Qt 界面与控制器（不写业务算法）
- `count_app/`: 动作计数Qt页面 




