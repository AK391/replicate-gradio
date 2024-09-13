# Installation

1. Clone this repo: `git clone git@github.com:gradio-app/replicate-gradio.git`
2. Navigate into the folder that you cloned this repo into: `cd replicate-gradio`
3. Install this package: `pip install -e .`

<!-- ```bash
pip install replicate-gradio
``` -->

That's it! When you import `replicate-gradio`, it will monkey-patch `gradio` to include a `.load_replicate` method that can be called to load and create GUIs around any Replicate API endpoint.

# Basic Usage

Just like if you were to use the `replicate` client, you should first save your Replicate API token to this environment variable:

```
export REPLICATE_API_TOKEN=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import replicate-gradio

gr.load_replicate(
    model='black-forest-labs/flux-schnell',
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to the model on Replicate!

<img width="1246" alt="image" src="https://github.com/user-attachments/assets/2c975cbd-965f-4967-9468-d791aabfc9aa">


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

with gr.Blocks() as demo:
    with gr.Tab("SDXL"):
        gr.load_replicate('replicate-ai/fast-sdxl)
    with gr.Tab("Flux"):
        gr.load_replicate('black-forest-labs/flux-schnell')

demo.launch()
```

# Under the Hood

The `replicate-gradio` Python library has two dependencies: `replicate` and `gradio`. When imported, the library monkey-patches `gradio` to add a `.load_replicate()` method that calls the Replicate Inference API.
