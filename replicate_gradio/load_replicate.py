import gradio as gr

def patch_gradio():
    gr.load_replicate = load_replicate

def load_replicate():
    return "replicate"