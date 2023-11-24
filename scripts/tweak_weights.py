"""
Tweak lora weights to determine init merger values
"""
import argparse
import gradio as gr
from diffusers import DiffusionPipeline
import torch
from ziplora_pytorch.utils import (
    get_lora_weights,
    merge_lora_weights,
    initialize_ziplora_layer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="pretrained model path",
    )
    parser.add_argument(
        "--lora_name_or_path", type=str, required=True, help="lora path"
    )
    parser.add_argument(
        "--lora_name_or_path_2", type=str, required=True, help="lora path"
    )
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=torch.float16
).to(device)
lora_weights = get_lora_weights(args.lora_name_or_path)
lora_weights_2 = get_lora_weights(args.lora_name_or_path_2)
current_w1 = 1.0
current_w2 = 1.0


def change_weights(unet, w1: float = 1.0, w2: float = 1.0):
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        merged_lora_weights_dict = merge_lora_weights(lora_weights, attn_name)
        merged_lora_weights_dict_2 = merge_lora_weights(lora_weights_2, attn_name)
        kwargs = {
            "state_dict": merged_lora_weights_dict,
            "state_dict_2": merged_lora_weights_dict_2,
        }
        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer(
                part="to_q",
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                init_merger_value=w1,
                init_merger_value_2=w2,
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer(
                part="to_k",
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                init_merger_value=w1,
                init_merger_value_2=w2,
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer(
                part="to_v",
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                init_merger_value=w1,
                init_merger_value_2=w2,
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer(
                part="to_out.0",
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                init_merger_value=w1,
                init_merger_value_2=w2,
                **kwargs,
            )
        )
    return unet


def run(
    prompt: str,
    lora1_weight: float = 1.0,
    lora2_weight: float = 1.0,
):
    global current_w1, current_w2
    if (current_w1, current_w2) != (lora1_weight, lora2_weight):
        pipe.unet = change_weights(pipe.unet, lora1_weight, lora2_weight)
        current_w1 = lora1_weight
        current_w2 = lora2_weight
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(prompt=prompt, generator=generator).images[0]
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(label="prompt", value="a sbu dog in szn style")
            w1 = gr.Number(label="lora weight 1")
            w2 = gr.Number(label="lora weight 2")
            bttn = gr.Button(value="Run")
        with gr.Column():
            out = gr.Image(label="out")
    prompt.submit(fn=run, inputs=[prompt, w1, w2], outputs=[out])
    bttn.click(fn=run, inputs=[prompt, w1, w2], outputs=[out])
    demo.launch(share=True, debug=True, show_error=True)
