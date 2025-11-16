import argparse
import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
from huggingface_hub import login

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_PROMPT = (
    "portrait of a young woman, soft natural lighting, hyperrealistic skin, detailed hands, detailed eyes"
)
DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, (low quality, worst quality:1.2), cartoon, anime, 3d, painting, sketch, "
    "mutated hands, extra fingers, missing fingers, extra limbs, bad anatomy"
)
SEED = -1
GUIDANCE_SCALE = 7.0
NUM_INFERENCE_STEPS = 28
OUTPUT_PATH = "output_img2img.png"
INIT_IMAGE_PATH = "/content/input_image.png"  # ← Change this or pass via CLI
STRENGTH = 0.6  # 0.0 = original, 1.0 = full rewrite

# -----------------------------
# Setup & Model Loading
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LORA_PATH = "/content/NSFW-Real/Hand v2.safetensors"

try:
    # 🔁 Use Img2Img pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "John6666/areal-noob-realistic-photography-cosplay-portrait-nsfw-beta-03-test16-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True,
        use_auth_token=HF_TOKEN
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Load and fuse LoRA
    pipe.load_lora_weights(LORA_PATH)
    pipe.fuse_lora(lora_scale=0.8)

    # Dtype consistency
    for module in [pipe.text_encoder, pipe.text_encoder_2, pipe.vae, pipe.unet]:
        if module is not None:
            module.to(torch.float16)

    # Compel for long prompts
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False
    )
    print("✅ Img2Img pipeline + LoRA + long prompts ready.")
except Exception as e:
    print(f"❌ Failed to load model or LoRA: {e}")
    exit(1)

# -----------------------------
# Long Prompt Handling (same as before)
# -----------------------------
def process_long_prompt(prompt, negative_prompt=""):
    try:
        conditioning, pooled = compel([prompt, negative_prompt])
        return conditioning, pooled
    except Exception as e:
        print(f"⚠️ Long prompt processing failed: {e}")
        return None, None

# -----------------------------
# Img2Img Inference
# -----------------------------
def img2img(
    init_image,
    prompt,
    negative_prompt,
    strength=0.6,
    seed=None,
    guidance_scale=7.0,
    num_inference_steps=28
):
    if seed is None or seed == -1:
        seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device=device).manual_seed(seed)

    use_long_prompt = len(prompt.split()) > 60 or len(prompt) > 300

    try:
        if use_long_prompt:
            print("🔍 Using long-prompt processing...")
            conditioning, pooled = process_long_prompt(prompt, negative_prompt)
            if conditioning is not None:
                output = pipe(
                    image=init_image,
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
                return output, seed

        print("🔤 Using standard prompt...")
        output = pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        return output, seed

    except Exception as e:
        print(f"💥 Error: {e}")
        return Image.new('RGB', init_image.size, color='black'), seed

# -----------------------------
# CLI Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SDXL Img2Img with LoRA")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--init_image", type=str, default=INIT_IMAGE_PATH)
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    return parser.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    # Load input image
    try:
        init_image = Image.open(args.init_image).convert("RGB")
        # Resize if needed (optional)
        # init_image = init_image.resize((1024, 1024))
    except Exception as e:
        print(f"❌ Failed to load input image: {e}")
        exit(1)

    image, used_seed = img2img(
        init_image=init_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        seed=None if args.seed == -1 else args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"✅ Saved to: {args.output}")

    try:
        from IPython.display import display
        display(image)
    except:
        pass