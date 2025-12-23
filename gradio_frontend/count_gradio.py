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
                        gr.Markdown("选择待评测视频，或启用摄像头实时评测，设置参数后开始评测")
                        with gr.Row():
                            with gr.Column(scale=1):
                                eval_video = gr.Video(label="待评测视频", interactive=True, height=280)
                        # 新增：启用摄像头切换
                        webcam_toggle = gr.Checkbox(label="启用摄像头", value=False)
                        # 新增：实时预览摄像头画面（仅用于回显，不可编辑）
                        webcam_preview = gr.Image(label="实时画面预览", interactive=False)

                        def on_toggle_webcam(enabled: bool):
                            # 启用摄像头时禁用并清空视频输入
                            if enabled:
                                return gr.update(interactive=False, value=None)
                            return gr.update(interactive=True)

                        webcam_toggle.change(
                            fn=on_toggle_webcam,
                            inputs=[webcam_toggle],
                            outputs=[eval_video],
                        )

                        with gr.Row():
                            tol = gr.Slider(1, 30, value=10, step=1, label="模板角度容差(°)")
                            key_actions = gr.Number(value=4, precision=0, label="关键动作数")
                        with gr.Row():
                            eval_start_btn = gr.Button("开始评测", variant="primary")
                            eval_stop_btn = gr.Button("停止评测", variant="secondary")
                            eval_clear_btn = gr.Button("清空")
                        eval_progress_text = gr.Textbox(label="评测计数与进度(实时)", interactive=False, lines=8, max_lines=16)

                        def on_start_eval(eval_file: Optional[str], tolerance: float, key_cnt: float, use_webcam: bool):
                            k = None if key_cnt is None else int(key_cnt)
                            # 摄像头模式：忽略文件输入，同时回显每帧画面
                            if use_webcam:
                                for txt, frame in controller.start_template_evaluation(None, True, tolerance, k):
                                    yield txt, frame
                                return
                            # 文件模式：需要上传
                            if not eval_file:
                                yield "请上传待评测视频后再开始。", None
                                return
                            for txt, frame in controller.start_template_evaluation(eval_file, False, tolerance, k):
                                yield txt, frame

                        eval_start_btn.click(
                            fn=on_start_eval,
                            inputs=[eval_video, tol, key_actions, webcam_toggle],
                            outputs=[eval_progress_text, webcam_preview],
                        )

                        def on_stop_eval():
                            controller.stop()
                            # 给出简短提示，实际停止由生成器循环检查标志实现
                            return "已请求停止评测（摄像头模式将在下一帧停止）"

                        eval_stop_btn.click(
                            fn=on_stop_eval,
                            inputs=[],
                            outputs=[eval_progress_text],
                        )

                        def on_clear_eval():
                            # 清空视频、参数、摄像头开关以及预览和文本
                            controller.stop()
                            return None, 10, 4, False, "", None

                        eval_clear_btn.click(
                            fn=on_clear_eval,
                            inputs=[],
                            outputs=[eval_video, tol, key_actions, webcam_toggle, eval_progress_text, webcam_preview],
                        )

                        def on_play_event(video_path: Optional[str], use_webcam: bool):
                            # 启用摄像头时不响应视频播放，提示使用“开始评测”
                            if use_webcam:
                                yield "当前已启用摄像头，请点击『开始评测』以启动实时处理。"
                                return
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
                            inputs=[eval_video, webcam_toggle],
                            outputs=[eval_progress_text],
                        )

        with gr.Accordion("提示", open=False):
            gr.Markdown(
                """
                - 若关键动作数填0，系统将采用固定前瞻4帧策略。
                - 文本框只显示计数与进度，不展示处理后的图像帧。
                - 可勾选“启用摄像头”使用实时视频流进行评测。
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