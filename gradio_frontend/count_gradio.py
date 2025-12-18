from __future__ import annotations

import os
from typing import Optional

import gradio as gr

from gradio_frontend.mvc.controller import CountController

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".temp")
os.makedirs(TEMP_DIR, exist_ok=True)
# 将 Gradio 的临时目录强制指向项目 .temp，避免使用系统临时目录
os.environ.setdefault("GRADIO_TEMP_DIR", TEMP_DIR)

# 视图仅负责构建UI与绑定交互

def build_ui() -> gr.Blocks:
    controller = CountController()
    with gr.Blocks(title="动作计数演示（两阶段）") as demo:
        gr.Markdown(
            "## 流程\n1. 先上传参考视频并处理，系统生成角度模板。\n2. 然后上传待评测视频或勾选摄像头，填写关键动作数与容差，开始实时计数。\n\n注：不再展示处理后的视频帧，仅显示计数与进度文本。"
        )
        # 步骤1：参考视频处理
        with gr.Tab("步骤1：参考视频"):
            ref_video = gr.Video(label="参考视频")
            ref_start_btn = gr.Button("处理参考视频")
            ref_info_text = gr.Textbox(label="参考模板信息与进度", interactive=False)

            def on_start_ref(ref_file: Optional[str]):
                for r_txt, _ in controller.start_reference(ref_file):
                    # 仅输出文本信息，不输出图片
                    yield r_txt

            ref_start_btn.click(
                fn=on_start_ref,
                inputs=[ref_video],
                outputs=[ref_info_text],
            )

        # 步骤2：评测（只显示计数与进度文本）
        with gr.Tab("步骤2：评测"):
            with gr.Row():
                eval_video = gr.Video(label="待评测视频")
                use_cam = gr.Checkbox(label="使用摄像头", value=False)
            with gr.Row():
                tol = gr.Slider(1, 30, value=10, step=1, label="模板角度容差(°)")
                key_actions = gr.Number(value=4, precision=0, label="关键动作数")
            eval_start_btn = gr.Button("开始评测")
            eval_progress_text = gr.Textbox(label="评测计数与进度(实时)", interactive=False)

            def on_start_eval(eval_file: Optional[str], use_cam_flag: bool, tolerance: float, key_cnt: float):
                # 若为0则传0，控制器将采用固定前瞻4帧；None表示未设置
                k = None if key_cnt is None else int(key_cnt)
                for txt, _ in controller.start_template_evaluation(eval_file, use_cam_flag, tolerance, k):
                    yield txt

            eval_start_btn.click(
                fn=on_start_eval,
                inputs=[eval_video, use_cam, tol, key_actions],
                outputs=[eval_progress_text],
            )

            # 新增：绑定“视频播放”事件，开始计数
            def on_play_event(video_path: Optional[str]):
                # 使用当前UI上的参数：保留0值
                k_val = key_actions.value
                k = None if k_val is None else int(k_val)
                tolerance = float(tol.value)
                # 注意：播放事件不支持摄像头，这里仅使用视频文件
                for txt, _ in controller.start_template_evaluation(video_path, use_webcam=False, tolerance_deg=tolerance, key_actions=k):
                    yield txt

            eval_video.play(
                fn=on_play_event,
                inputs=[eval_video],
                outputs=[eval_progress_text],
            )
    return demo


def launch(server_name: str | None = None, server_port: int | None = None):
    demo = build_ui()
    demo.launch(server_name=server_name, server_port=server_port)

if __name__ =="__main__":
    launch("localhost",10620)
