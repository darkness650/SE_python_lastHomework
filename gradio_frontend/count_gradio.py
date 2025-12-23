from __future__ import annotations

import os
from typing import Optional

import gradio as gr
from gradio import themes

from gradio_frontend.mvc.controller import CountController

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", TEMP_DIR)

PAGE_CSS = """
/* 全局将次级背景改为白色，避免灰色区域突兀 */
:root { --color-background-secondary: #ffffff; }
/* 容器背景也保持白色 */
.gradio-container { background: #ffffff; }
/* Video/Canvas 默认背景置白，以免出现灰底 */
video, canvas { background: #ffffff !important; }
/* 步骤1与步骤2之间添加垂直分割线 */
#step1_col { border-right: 1px solid #e5e7eb; padding-right: 12px; }
#step2_col { padding-left: 12px; }
@media (max-width: 768px) {
    /* 移动端去掉分割线，避免拥挤 */
    #step1_col { border-right: none; padding-right: 0; }
    #step2_col { padding-left: 0; }
}
"""

def build_ui() -> gr.Blocks:
    controller = CountController()
    with gr.Blocks(
        title="动作计数演示（两阶段）",
    ) as demo:
        gr.Markdown(
            """
            ## 流程
            <div style=\"font-size:1.1em; line-height:1.8;\">
            1. 先上传参考视频并处理，系统生成角度模板。<br>
            2. 然后上传待评测视频，填写关键动作数与容差，开始实时计数。<br>
            注：不再展示处理后的视频帧，仅显示计数与进度文本。
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_id="step1_col"):
                with gr.Tab("步骤1：参考视频"):
                    with gr.Group():
                        gr.Markdown("上传参考视频并生成角度模板")
                        ref_video = gr.Video(label="参考视频", interactive=True, height=280)
                        with gr.Row():
                            ref_start_btn = gr.Button("处理参考视频", variant="primary")
                            ref_clear_btn = gr.Button("清空")
                        ref_info_text = gr.Textbox(label="参考模板信息与进度", interactive=False, lines=6, max_lines=12, show_label=True)

                        def on_start_ref(ref_file: Optional[str]):
                            if not ref_file:
                                yield "请先上传参考视频再点击处理。"
                                return
                            for r_txt, _ in controller.start_reference(ref_file):
                                yield r_txt

                        ref_start_btn.click(
                            fn=on_start_ref,
                            inputs=[ref_video],
                            outputs=[ref_info_text],
                        )

                        def on_clear_ref():
                            return None, ""

                        ref_clear_btn.click(
                            fn=on_clear_ref,
                            inputs=[],
                            outputs=[ref_video, ref_info_text],
                        )

            with gr.Column(scale=1, elem_id="step2_col"):
                with gr.Tab("步骤2：评测"):
                    with gr.Group():
                        gr.Markdown("选择待评测视频，设置参数后开始评测")
                        with gr.Row():
                            with gr.Column(scale=1):
                                eval_video = gr.Video(label="待评测视频", interactive=True, height=280)
                        with gr.Row():
                            tol = gr.Slider(1, 30, value=10, step=1, label="模板角度容差(°)")
                            key_actions = gr.Number(value=4, precision=0, label="关键动作数")
                        with gr.Row():
                            eval_start_btn = gr.Button("开始评测", variant="primary")
                            eval_clear_btn = gr.Button("清空")
                        eval_progress_text = gr.Textbox(label="评测计数与进度(实时)", interactive=False, lines=8, max_lines=16)

                        def on_start_eval(eval_file: Optional[str], tolerance: float, key_cnt: float):
                            k = None if key_cnt is None else int(key_cnt)
                            if not eval_file:
                                yield "请上传待评测视频后再开始。"
                                return
                            for txt, _ in controller.start_template_evaluation(eval_file, False, tolerance, k):
                                yield txt

                        eval_start_btn.click(
                            fn=on_start_eval,
                            inputs=[eval_video, tol, key_actions],
                            outputs=[eval_progress_text],
                        )

                        def on_clear_eval():
                            return None, 10, 4, ""

                        eval_clear_btn.click(
                            fn=on_clear_eval,
                            inputs=[],
                            outputs=[eval_video, tol, key_actions, eval_progress_text],
                        )

                        def on_play_event(video_path: Optional[str]):
                            k_val = key_actions.value
                            k = None if k_val is None else int(k_val)
                            tolerance = float(tol.value)
                            if not video_path:
                                yield "请先选择视频文件后再播放进行评测。"
                                return
                            for txt, _ in controller.start_template_evaluation(video_path, use_webcam=False, tolerance_deg=tolerance, key_actions=k):
                                yield txt

                        eval_video.play(
                            fn=on_play_event,
                            inputs=[eval_video],
                            outputs=[eval_progress_text],
                        )

        with gr.Accordion("提示", open=False):
            gr.Markdown(
                """
                - 若关键动作数填0，系统将采用固定前瞻4帧策略。
                - 文本框只显示计数与进度，不展示处理后的图像帧。
                """
            )
    return demo


def launch(server_name: str | None = None, server_port: int | None = None):
    demo = build_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        theme=themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=PAGE_CSS,
    )

if __name__ =="__main__":
    launch("localhost", 10621)
