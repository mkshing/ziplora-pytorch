import fire
import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_pytorch.utils import insert_ziplora_to_unet


def main(
    ziplora_name_or_path: str,
    prompt: str,
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
):
    pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path)
    pipeline.unet = insert_ziplora_to_unet(pipeline.unet, ziplora_name_or_path)
    pipeline.to(device="cuda", dtype=torch.float16)
    image = pipeline(prompt=prompt).images[0]
    image.save("out.png")


if __name__ == "__main__":
    fire.Fire(main)
