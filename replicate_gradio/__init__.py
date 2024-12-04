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
    """Convert bytes to PIL Image and ensure we get fresh data"""
    if isinstance(byte_data, bytes):
        return Image.open(io.BytesIO(byte_data))
    # For file-like objects
    if hasattr(byte_data, 'seek'):
        byte_data.seek(0)
    return Image.open(io.BytesIO(byte_data.read()))

PIPELINE_REGISTRY = {
    "text-to-image": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("width", gr.Number, {"label": "Width", "value": 1024, "minimum": 512, "maximum": 2048, "step": 64, "optional": True}),
            ("height", gr.Number, {"label": "Height", "value": 1024, "minimum": 512, "maximum": 2048, "step": 64, "optional": True}),
            ("num_outputs", gr.Number, {"label": "Number of Images", "value": 1, "minimum": 1, "maximum": 4, "step": 1, "optional": True}),
            ("scheduler", gr.Dropdown, {"label": "Scheduler", "choices": ["DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras"], "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 30, "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 7.5, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("images", gr.Gallery, {})],
        "preprocess": lambda *args: {
            k: v for k, v in zip([
                "prompt", "negative_prompt", "width", "height", "num_outputs",
                "scheduler", "num_inference_steps", "guidance_scale", "seed"
            ], args) if v is not None and v != ""
        },
        "postprocess": lambda x: [bytes_to_image(img) for img in x] if isinstance(x, list) else [bytes_to_image(x)]
    },

    "image-to-image": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("image", gr.Image, {"label": "Input Image", "type": "pil"}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("strength", gr.Slider, {"label": "Strength", "minimum": 0, "maximum": 1, "value": 0.7, "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 30, "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 7.5, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("images", gr.Gallery, {})],
        "preprocess": lambda *args: {
            k: (resize_image_if_needed(v) if k == "image" else v)
            for k, v in zip([
                "prompt", "image", "negative_prompt", "strength",
                "num_inference_steps", "guidance_scale", "seed"
            ], args) if v is not None and v != ""
        },
        "postprocess": lambda x: [bytes_to_image(img) for img in x] if isinstance(x, list) else [bytes_to_image(x)]
    },

    "control-net": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("control_image", gr.Image, {"label": "Control Image", "type": "pil"}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 7.5, "optional": True}),
            ("control_guidance_scale", gr.Slider, {"label": "Control Guidance Scale", "minimum": 1, "maximum": 20, "value": 1.5, "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 30, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("images", gr.Gallery, {})],
        "preprocess": lambda *args: {
            k: (resize_image_if_needed(v) if k == "control_image" else v)
            for k, v in zip([
                "prompt", "control_image", "negative_prompt", "guidance_scale",
                "control_guidance_scale", "num_inference_steps", "seed"
            ], args) if v is not None and v != ""
        },
        "postprocess": lambda x: [bytes_to_image(img) for img in x] if isinstance(x, list) else [bytes_to_image(x)]
    },

    "inpainting": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("image", gr.Image, {"label": "Original Image", "type": "pil"}),
            ("mask", gr.Image, {"label": "Mask Image", "type": "pil"}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 30, "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 7.5, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("images", gr.Gallery, {})],
        "preprocess": lambda *args: {
            k: (resize_image_if_needed(v) if k in ["image", "mask"] else v)
            for k, v in zip([
                "prompt", "image", "mask", "negative_prompt",
                "num_inference_steps", "guidance_scale", "seed"
            ], args) if v is not None and v != ""
        },
        "postprocess": lambda x: [bytes_to_image(img) for img in x] if isinstance(x, list) else [bytes_to_image(x)]
    },

    "text-to-video": {
        "inputs": [
            ("prompt", gr.Textbox, {"label": "Prompt"}),
            ("negative_prompt", gr.Textbox, {"label": "Negative Prompt", "optional": True}),
            ("num_frames", gr.Number, {"label": "Number of Frames", "value": 16, "minimum": 14, "maximum": 120, "step": 1, "optional": True}),
            ("fps", gr.Number, {"label": "FPS", "value": 8, "minimum": 1, "maximum": 30, "step": 1, "optional": True}),
            ("num_inference_steps", gr.Slider, {"label": "Steps", "minimum": 1, "maximum": 100, "value": 50, "optional": True}),
            ("guidance_scale", gr.Slider, {"label": "Guidance Scale", "minimum": 1, "maximum": 20, "value": 9.0, "optional": True}),
            ("width", gr.Number, {"label": "Width", "value": 576, "minimum": 320, "maximum": 1024, "step": 64, "optional": True}),
            ("height", gr.Number, {"label": "Height", "value": 320, "minimum": 320, "maximum": 576, "step": 64, "optional": True}),
            ("seed", gr.Number, {"label": "Seed", "optional": True})
        ],
        "outputs": [("video", gr.Video, {})],
        "preprocess": lambda *args: {
            k: v for k, v in zip([
                "prompt", "negative_prompt", "num_frames", "fps",
                "num_inference_steps", "guidance_scale", "width", "height", "seed"
            ], args) if v is not None and v != ""
        },
        "postprocess": lambda x: x
    },
}

MODEL_TO_PIPELINE = {
    "stability-ai/sdxl": "text-to-image",
    "black-forest-labs/flux-pro": "text-to-image",
    "stability-ai/stable-diffusion": "text-to-image",
    
    "black-forest-labs/flux-depth-pro": "control-net",
    "black-forest-labs/flux-canny-pro": "control-net",
    "black-forest-labs/flux-depth-dev": "control-net",
    
    "black-forest-labs/flux-fill-pro": "inpainting",
    "stability-ai/stable-diffusion-inpainting": "inpainting",
    "zsxkib/hunyuan-video:349dbe0feb6e8e4a6fab3c6a4dd642413e6c10735353de8b40f12abeee203617": "text-to-video",
}

def create_component(comp_type: type, name: str, config: Dict[str, Any]) -> gr.components.Component:
    # Remove 'optional' from config as it's not a valid Gradio parameter
    config = config.copy()
    is_optional = config.pop('optional', False)
    
    # Add "(Optional)" to label if the field is optional
    if is_optional:
        label = config.get('label', name)
        config['label'] = f"{label} (Optional)"
    
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

async def async_run_with_timeout(model_name: str, args: dict, save_path: str = None):
    try:
        output = replicate.run(
            model_name,
            input=args
        )
        
        # If save_path is provided and output is file-like, save to disk
        if save_path and hasattr(output, 'read'):
            with open(save_path, "wb") as file:
                file.write(output.read())
            return save_path
            
        return output
    except Exception as e:
        raise gr.Error(f"Model prediction failed: {str(e)}")

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable):
    async def fn(*args):
        args = preprocess(*args)
        outputs = await async_run_with_timeout(model_name, args)
        # Force immediate processing of outputs
        if isinstance(outputs, list):
            outputs = [output.read() if hasattr(output, 'read') else output for output in outputs]
        elif hasattr(outputs, 'read'):
            outputs = outputs.read()
        return postprocess(outputs)
    return fn

def registry(name: str | Dict, token: str | None = None, inputs=None, outputs=None, src=None, accept_token: bool = False, **kwargs) -> gr.Interface:
    """
    Create a Gradio Interface for a model on Replicate.
    Parameters:
        - name (str | Dict): The name of the model on Replicate, or a dict with model info.
        - token (str, optional): The API token for the model on Replicate.
        - inputs (List[gr.Component], optional): The input components to use instead of the default.
        - outputs (List[gr.Component], optional): The output components to use instead of the default.
        - src (callable, optional): Ignored, used by gr.load for routing.
        - accept_token (bool, optional): Whether to accept a token input field.
    Returns:
        gr.Interface: A Gradio interface for the model.
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
    
    # Add token input if accept_token is True
    if accept_token:
        token_input = gr.Textbox(label="API Token", type="password")
        inputs_ = [token_input] + inputs_
        
        # Modify preprocess function to handle token
        original_preprocess = preprocess
        def new_preprocess(token, *args):
            if token:
                os.environ["REPLICATE_API_TOKEN"] = token
            return original_preprocess(*args)
        preprocess = new_preprocess
    
    inputs, outputs = inputs or inputs_, outputs or outputs_

    fn = get_fn(model_name, preprocess, postprocess)
    return gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)


__version__ = "0.1.0"
