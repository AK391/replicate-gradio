import gradio as gr
import replicate_gradio



# Create and launch interface
demo = gr.load(
    name='zsxkib/hunyuan-video:349dbe0feb6e8e4a6fab3c6a4dd642413e6c10735353de8b40f12abeee203617',
    src=replicate_gradio.registry
).launch()
