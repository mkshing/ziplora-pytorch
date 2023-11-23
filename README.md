# ZipLoRA-pytorch
This is an implementation of [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](https://ziplora.github.io/) by using [🤗diffusers](https://github.com/huggingface/diffusers).


## Installation
```
git clone git@github.com:mkshing/ziplora-pytorch.git
cd ziplora-pytorch
pip install -r requirements.txt
```

## Usage

### 1. Train LoRAs for subject/style images
In this step, 2 LoRAs for subject/style images are trained based on SDXL. Using SDXL here is important because they found that the pre-trained SDXL exhibits strong learning when fine-tuned on only one reference style image.

Fortunately, diffusers already implemented LoRA based on SDXL [here](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md) and you can simply follow the instruction. 

For example, your training script would be like this.
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# for subject
export OUTPUT_DIR="lora-sdxl-dog"
export INSTANCE_DIR="dog"
export PROMPT="a sks dog"
export VALID_PROMPT="a sks dog in a bucket"

# for style
# export OUTPUT_DIR="lora-sdxl-waterpainting"
# export INSTANCE_DIR="waterpainting"
# export PROMPT="a cat of in szn style"
# export VALID_PROMPT="a man in szn style"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=64 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=50 \
  --seed="0" \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --push_to_hub \
```

### 2. Train ZipLoRA

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# for subject
export LORA_PATH="mkshing/lora-sdxl-dog"
export INSTANCE_DIR="dog"
export PROMPT="a sks dog"

# for style
export LORA_PATH2="mkshing/lora-sdxl-waterpainting"
export INSTANCE_DIR2="waterpainting"
export PROMPT2="a cat of in szn style"

# general 
export OUTPUT_DIR="ziplora-sdxl-dog-waterpainting"
export VALID_PROMPT="a sks dog in szn style"


accelerate launch train_dreambooth_ziplora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --lora_name_or_path=$LORA_PATH \
  --instance_prompt="${PROMPT}" \
  --instance_data_dir=$INSTANCE_DIR \
  --lora_name_or_path_2=$LORA_PATH2 \
  --instance_prompt_2="${PROMPT2}" \
  --instance_data_dir_2=$INSTANCE_DIR2 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --similarity_lambda=0.01 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=10 \
  --seed="0" \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \

```

### 3. Inference

```python
import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_pytorch.utils import insert_ziplora_to_unet

pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path)
pipeline.unet = insert_ziplora_to_unet(pipeline.unet, ziplora_name_or_path)
pipeline.to(device="cuda", dtype=torch.float16)
image = pipeline(prompt=prompt).images[0]
image.save("out.png")
```


## TODO

- [x] super quick instruction for training each loras
- [x] ZipLoRA (training)
- [x] ZipLoRA (inference)
- [ ] Investigate initial values of mergers 