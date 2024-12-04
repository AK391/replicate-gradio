import gradio as gr
import replicate_gradio



# Create and launch interface
demo = gr.load(
    name='zsxkib/hunyuan-video',
    src=replicate_gradio.registry
).launch()
