import gradio as gr
import replicate
import asyncio
import os
from typing import Callable, Dict, Any, List, Tuple
import httpx
from PIL import Image
import io
import base64
import numpy as np

def resize_image_if_needed(image, max_size=1024):
    """Resize image if either dimension exceeds max_size while maintaining aspect ratio"""
    if isinstance(image, str) and image.startswith('data:image'):
        return image  # Already a data URI, skip processing
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Get original dimensions
    width, height = image.size
    
    # Calculate new dimensions if needed
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def bytes_to_image(byte_data):
    """Convert bytes to PIL Image"""
    return Image.open(io.BytesIO(byte_data))

PIPELINE_REGISTRY = {
    "depth-control": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("control_image", gr.Image, {"label": "Control Image"}),
            ("guidance", gr.Slider, {"minimum": 1, "maximum": 20, "value": 7, "label": "Guidance"})
        ],
        "outputs": [("image", gr.Image, {})],
        "preprocess": lambda prompt, control_image, guidance: {
            "prompt": prompt,
            "control_image": resize_image_if_needed(control_image),
            "guidance": guidance
        },
        "postprocess": lambda x: bytes_to_image(x[0].read() if isinstance(x, list) else x.read())
    },
    "canny-control": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("control_image", gr.Image, {"label": "Control Image"}),
            ("steps", gr.Slider, {"minimum": 1, "maximum": 50, "value": 28, "label": "Steps"}),
            ("guidance", gr.Slider, {"minimum": 1, "maximum": 50, "value": 25, "label": "Guidance"})
        ],
        "outputs": [("image", gr.Image, {})],
        "preprocess": lambda prompt, control_image, steps, guidance: {
            "prompt": prompt,
            "control_image": resize_image_if_needed(control_image),
            "steps": steps,
            "guidance": guidance
        },
        "postprocess": lambda x: x[0].read() if isinstance(x, list) else x.read()
    },
    "inpainting": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("image", gr.Image, {"label": "Original Image"}),
            ("mask", gr.Image, {"label": "Mask Image", "tool": "sketch", "source": "canvas", "type": "numpy"}),
        ],
        "outputs": [("image", gr.Image, {})],
        "preprocess": lambda prompt, image, mask: {
            "prompt": prompt,
            "image": resize_image_if_needed(image),
            "mask": resize_image_if_needed(mask),
        },
        "postprocess": lambda x: x[0].read() if isinstance(x, list) else x.read()
    },
    "depth-dev": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("control_image", gr.Image, {"label": "Control Image"}),
        ],
        "outputs": [("image", gr.Gallery, {})],
        "preprocess": lambda prompt, control_image: {
            "prompt": prompt,
            "control_image": resize_image_if_needed(control_image)
        },
        "postprocess": lambda x: [img.read() for img in x] if isinstance(x, list) else x.read()
    }
}

MODEL_TO_PIPELINE = {
    "black-forest-labs/flux-depth-pro": "depth-control",
    "black-forest-labs/flux-canny-pro": "canny-control",
    "black-forest-labs/flux-fill-pro": "inpainting",
    "black-forest-labs/flux-depth-dev": "depth-dev"
}

def create_component(comp_type: type, name: str, config: Dict[str, Any]) -> gr.components.Component:
    return comp_type(label=config.get("label", name), **{k:v for k,v in config.items() if k != "label"})

def get_pipeline(model: str) -> str:
    return MODEL_TO_PIPELINE.get(model, "text-to-image")

def get_interface_args(pipeline: str) -> Tuple[List, List, Callable, Callable]:
    if pipeline not in PIPELINE_REGISTRY:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    
    config = PIPELINE_REGISTRY[pipeline]
    
    inputs = [create_component(comp_type, name, conf) 
             for name, comp_type, conf in config["inputs"]]
    
    outputs = [create_component(comp_type, name, conf) 
              for name, comp_type, conf in config["outputs"]]
    
    return inputs, outputs, config["preprocess"], config["postprocess"]

async def async_run_with_timeout(model_name: str, args: dict):
    try:
        output = replicate.run(
            model_name,
            input=args
        )
        return output
    except Exception as e:
        raise gr.Error(f"Model prediction failed: {str(e)}")

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable):
    async def fn(*args):
        args = preprocess(*args)
        outputs = await async_run_with_timeout(model_name, args)
        return postprocess(outputs)
    return fn

def registry(name: str | Dict, token: str | None = None, inputs=None, outputs=None, src=None, **kwargs) -> gr.Interface:
    """
    Create a Gradio Interface for a model on Replicate.
    Parameters:
        - name (str | Dict): The name of the model on Replicate, or a dict with model info.
        - token (str, optional): The API token for the model on Replicate.
        - inputs (List[gr.Component], optional): The input components to use instead of the default.
        - outputs (List[gr.Component], optional): The output components to use instead of the default.
        - src (callable, optional): Ignored, used by gr.load for routing.
    Returns:
        gr.Interface: A Gradio interface for the model.
    Example:
        ```python
        import gradio as gr
        import replicate_gradio

        gr.load(
            name='black-forest-labs/flux-depth-pro',
            src=replicate_gradio.registry
        ).launch()
        ```
    """
    # Handle both string names and dict configurations
    if isinstance(name, dict):
        model_name = name.get('name', name.get('model_name', ''))
    else:
        model_name = name

    if token:
        os.environ["REPLICATE_API_TOKEN"] = token
        
    pipeline = get_pipeline(model_name)
    inputs_, outputs_, preprocess, postprocess = get_interface_args(pipeline)
    inputs, outputs = inputs or inputs_, outputs or outputs_

    fn = get_fn(model_name, preprocess, postprocess)
    return gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

# Register the function with Gradio's loading system
gr.load = registry

__version__ = "0.1.0"
