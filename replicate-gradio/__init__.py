import gradio as gr
from load_replicate import load_replicate

def patch_gradio():
    """
    Add a load_replicate method to the Gradio module.
    """
    gr.load_replicate = load_replicate

__version__ = "0.1.0"
