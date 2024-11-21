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

gr.load_replicate(
    model='black-forest-labs/flux-schnell',
).launch()
```

# Specialized Model Support

The library includes built-in support for several specialized model types with custom interfaces:

- **Depth Control** (`black-forest-labs/flux-depth-pro`)
- **Canny Control** (`black-forest-labs/flux-canny-pro`) 
- **Inpainting** (`black-forest-labs/flux-fill-pro`)
- **Depth Development** (`black-forest-labs/flux-depth-dev`)

These models automatically configure appropriate input and output components. For example:

```python
import gradio as gr
import replicate_gradio

# Create a depth-control interface
gr.load_replicate(
    'black-forest-labs/flux-depth-pro',
).launch()
```

# Customization 

Once you can create a Gradio UI from a Replicate endpoint, you can customize it by setting your own input and output components, or any other arguments to `gr.Interface`. For example, the screenshot above was generated with:

```py
import gradio as gr
import replicate_gradio

gr.load_replicate(
    'black-forest-labs/flux-schnell',
    inputs=gr.Textbox(lines=4),
    examples=["a 19th century portrait of a man on the moon", "a small cartoon mouse eating an ice cream cone"],
).launch()
```


# Composition

Or use your loaded Interface within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import replicate_gradio

with gr.Blocks() as demo:
    with gr.Tab("SDXL"):
        gr.load_replicate('replicate-ai/fast-sdxl')
    with gr.Tab("Flux"):
        gr.load_replicate('black-forest-labs/flux-schnell')

demo.launch()
```

# Under the Hood

The `replicate-gradio` Python library has two dependencies: `replicate` and `gradio`. The library provides a custom loader that creates Gradio interfaces for Replicate models, with built-in support for specialized model types and custom interfaces.

# Authentication Options

There are two ways to provide your Replicate API token:

1. Environment variable (recommended):
```python
import os
os.environ["REPLICATE_API_TOKEN"] = "your-token-here"
```

2. Through the interface (useful for sharing demos):
```python
gr.load_replicate(
    'model-name',
    accept_token=True  # Adds a password field for the API token
).launch()
```

-------

Note: if you are getting a 401 authentication error, then the Replicate API Client is not able to get the API token from the environment variable. This happened to me as well, in which case save it in your Python session, like this:

```py
import os

os.environ["REPLICATE_API_TOKEN"] = ...
```