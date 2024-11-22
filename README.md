# Installation

1. Clone this repo: `git clone git@github.com:gradio-app/replicate-gradio.git`
2. Navigate into the folder that you cloned this repo into: `cd replicate-gradio`
3. Install this package: `pip install -e .`

<!-- ```bash
pip install replicate-gradio
``` -->

That's it! When you import `replicate-gradio`, it provides a `gr.load_replicate` method that can be used to create GUIs around any Replicate API endpoint.

# Basic Usage

Just like if you were to use the `replicate` client, you should first save your Replicate API token to this environment variable:

```bash
export REPLICATE_API_TOKEN=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import replicate_gradio

demo = gr.load(
    name='stability-ai/sdxl',  # or any other supported model
    src=replicate_gradio.registry
).launch()
```

# Supported Model Types

The library includes built-in support for several model pipelines with custom interfaces:

- **Text-to-Image**
  - `stability-ai/sdxl`
  - `black-forest-labs/flux-pro`
  - `stability-ai/stable-diffusion`
- **Control-Net Models**
  - `black-forest-labs/flux-depth-pro`
  - `black-forest-labs/flux-canny-pro`
  - `black-forest-labs/flux-depth-dev`
- **Inpainting Models**
  - `black-forest-labs/flux-fill-pro`
  - `stability-ai/stable-diffusion-inpainting`

Each pipeline type automatically configures appropriate input and output components. For example:

```python
import gradio as gr
import replicate_gradio

# Create a text-to-image interface
demo = gr.load(
    name='stability-ai/sdxl',
    src=replicate_gradio.registry
).launch()

# Create a control-net interface
demo = gr.load(
    name='black-forest-labs/flux-depth-pro',
    src=replicate_gradio.registry
).launch()

# Create an inpainting interface
demo = gr.load(
    name='black-forest-labs/flux-fill-pro',
    src=replicate_gradio.registry
).launch()
```

# Customization 

You can customize the interface by providing additional arguments that work with `gr.Interface`. For example:

```python
import gradio as gr
import replicate_gradio

demo = gr.load(
    name='stability-ai/sdxl',
    src=replicate_gradio.registry,
    inputs=gr.Textbox(lines=4),
    examples=["a 19th century portrait of a man on the moon", "a small cartoon mouse eating an ice cream cone"],
).launch()
```

# Composition

You can use the interfaces within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import replicate_gradio

with gr.Blocks() as demo:
    with gr.Tab("SDXL"):
        gr.load(
            name='stability-ai/sdxl',
            src=replicate_gradio.registry
        )
    with gr.Tab("Flux"):
        gr.load(
            name='black-forest-labs/flux-pro',
            src=replicate_gradio.registry
        )

demo.launch()
```

# Authentication Options

There are three ways to provide your Replicate API token:

1. Environment variable (recommended):
```bash
export REPLICATE_API_TOKEN=<your-token>
```

2. Direct in Python:
```python
import os
os.environ["REPLICATE_API_TOKEN"] = "your-token-here"
```

3. Through the interface (useful for sharing demos):
```python
demo = gr.load(
    name='model-name',
    src=replicate_gradio.registry,
    accept_token=True  # Adds a password field for the API token
).launch()
```

# Under the Hood

The `replicate-gradio` Python library has two dependencies: `replicate` and `gradio`. The library provides a custom loader that creates Gradio interfaces for Replicate models, with built-in support for specialized model types and custom interfaces.

-------

Note: if you are getting a 401 authentication error, then the Replicate API Client is not able to get the API token from the environment variable. This happened to me as well, in which case save it in your Python session, like this:

```py
import os

os.environ["REPLICATE_API_TOKEN"] = ...
```