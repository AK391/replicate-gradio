import gradio as gr
import replicate_gradio



# Create and launch interface
demo = gr.load(
    name='black-forest-labs/flux-canny-pro',
    src=replicate_gradio.registry
).launch()
