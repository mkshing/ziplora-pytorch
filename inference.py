import argparse
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_pytorch.utils import insert_ziplora_to_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="pretrained model path",
    )
    parser.add_argument(
        "--ziplora_name_or_path", type=str, required=True, help="ziplora path"
    )
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path)
pipeline.unet = insert_ziplora_to_unet(pipeline.unet, args.ziplora_name_or_path)
pipeline.to(device=device, dtype=torch.float16)


def run(prompt: str):
    # generator = torch.Generator(device="cuda").manual_seed(42)
    generator = None
    image = pipeline(prompt=prompt, generator=generator).images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(label="prompt", value="a sbu dog in szn style")
            bttn = gr.Button(value="Run")
        with gr.Column():
            out = gr.Image(label="out")
    prompt.submit(fn=run, inputs=[prompt], outputs=[out])
    bttn.click(fn=run, inputs=[prompt], outputs=[out])

    demo.launch(share=True, debug=True, show_error=True)
