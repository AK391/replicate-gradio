import gradio as gr
import replicate

from typing import Callable

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable):
    def fn(*args):
        args = preprocess(*args)
        outputs = replicate.run(model_name, args)
        return postprocess(outputs)
    return fn

# Implemented manually for now, but could we parse the openapi spec to get the interface args?
def get_interface_args(pipeline):
    if pipeline == "text-to-image":
        inputs = [gr.Textbox()]
        outputs = [gr.Image()]
        preprocess = lambda x: {"prompt": x}
        postprocess = lambda x: x[0]
    return inputs, outputs, preprocess, postprocess

def get_pipeline(model):
    pipeline = "text-to-image"
    return pipeline

def load_replicate(model_name: str, inputs=None, outputs=None, **kwargs) -> gr.Interface:
    pipeline = get_pipeline(model_name)
    inputs_, outputs_, preprocess, postprocess = get_interface_args(pipeline)
    inputs, outputs = inputs or inputs_, outputs or outputs_

    # construct a gr.Interface object
    fn = get_fn(model_name, preprocess, postprocess)
    return gr.Interface(fn, inputs, outputs, **kwargs)

gr.load_replicate = load_replicate

__version__ = "0.1.0"
