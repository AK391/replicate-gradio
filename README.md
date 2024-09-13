# Installation

```bash
pip install replicate-gradio
```

That's it! Installing `replicate-gradio` will monkey-patch `gradio` to include a `.load_replicate` method that can be called to load and create GUIs around any Replicate API endpoint.

# Usage

## Basic Usage

```python
import gradio as gr

gr.load_replicate(
    model='replicate-ai/fast-sdxl',
    pipeline="text-to-image",
    api_key=REPLICATE_KEY
).launch()
```

# Customization & Composition

Once you can create a Gradio UI from a Replicate endpoint, you can customize it, or use it within larger Gradio Web UIs, e.g.

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab("SDXL"):
        gr.load_replicate('replicate-ai/fast-sdxl', pipeline="text-to-image", inputs=gr.Textbox(lines=4))
    with gr.Tab("Flux"):
        io = gr.load_replicate('replicate-ai/flux/schnell', pipeline="text-to-image")

demo.launch()
```

# Under the Hood

The `replicate-gradio` Python library has two dependencies: `replicate-client` and `gradio`. When installed, the library patches `gradio` to add a `.load_replicate()` method that calls the Replicate Inference API.
```

Let me know if this works for you!
