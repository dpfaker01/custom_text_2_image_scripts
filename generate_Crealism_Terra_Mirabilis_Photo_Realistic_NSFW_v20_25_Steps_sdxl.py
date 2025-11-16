import argparse
import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
from huggingface_hub import login

# -----------------------------
# Configuration (Editable)
# -----------------------------
DEFAULT_PROMPT = (
    "portrait of a young woman, soft natural lighting, hyperrealistic skin, detailed hands, detailed eyes, 85mm lens, f/1.8, "
    "cinematic bokeh, photorealistic, 4k, high resolution, sharp focus, natural skin texture, perfect hand anatomy with five fingers, "
    "individual eyelashes, catchlights in eyes, subtle freckles"
)
DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, (low quality, worst quality:1.2), cartoon, anime, 3d, painting, drawing, sketch, "
    "watermark, text, blurry, signature, username, jpeg artifacts, "
    "mutated hands, poorly drawn hands, extra fingers, missing fingers, fused fingers, malformed hands, "
    "extra hands, extra arms, extra legs, extra limbs, cloned body parts, asymmetric hands, disfigured hands, "
    "bad anatomy, distorted limbs, duplicate arms, three hands, six fingers, unnatural pose"
)
SEED = -1
WIDTH = 1024
HEIGHT = 1024
GUIDANCE_SCALE = 7.0
NUM_INFERENCE_STEPS = 28
OUTPUT_PATH = "output_lora_v2.png"

# -----------------------------
# Setup & Model Loading
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔗 Updated LoRA path (note the space in filename — handled safely)
LORA_PATH = "/content/NSFW-Real/Hand v2.safetensors"

try:
    # Load base SDXL model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "John6666/crealism-terra-mirabilis-photo-realistic-nsfw-sdxl-v20-25-steps-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True,
        use_auth_token=HF_TOKEN
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # ✅ Load and fuse your hand-fix LoRA
    print(f"🧩 Loading LoRA from: {LORA_PATH}")
    pipe.load_lora_weights(LORA_PATH)
    pipe.fuse_lora(lora_scale=0.8)  # Adjust 0.6–1.0 as needed
    print("✅ LoRA fused into pipeline.")

    # Ensure dtype consistency
    for module in [pipe.text_encoder, pipe.text_encoder_2, pipe.vae, pipe.unet]:
        if module is not None:
            module.to(torch.float16)

    # Compel for long prompts — already supports unlimited length
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False  # ← Critical for long prompts
    )
    print("✅ Model, LoRA, and long-prompt support loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model or LoRA: {e}")
    exit(1)

# -----------------------------
# Long Prompt Handling
# -----------------------------
def process_long_prompt(prompt, negative_prompt=""):
    try:
        conditioning, pooled = compel([prompt, negative_prompt])
        return conditioning, pooled
    except Exception as e:
        print(f"⚠️ Long prompt processing failed: {e}, falling back to standard processing.")
        return None, None

# -----------------------------
# Inference Function
# -----------------------------
def generate_image(
    prompt,
    negative_prompt,
    seed=None,
    width=1024,
    height=1024,
    guidance_scale=7.0,
    num_inference_steps=28
):
    if seed is None or seed == -1:
        seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"🌱 Seed: {seed}")
    print(f"📝 Prompt length: {len(prompt)} chars ({len(prompt.split())} words)")
    # Trigger long-prompt path for detailed prompts
    use_long_prompt = len(prompt.split()) > 60 or len(prompt) > 300

    try:
        if use_long_prompt:
            print("🔍 Using long-prompt processing via Compel...")
            conditioning, pooled = process_long_prompt(prompt, negative_prompt)
            if conditioning is not None:
                output = pipe(
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
                return output, seed

        print("🔤 Using standard prompt processing...")
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        return output, seed

    except RuntimeError as e:
        print(f"💥 RuntimeError during generation: {e}")
        return Image.new('RGB', (width, height), color='black'), seed

# -----------------------------
# CLI Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with Hand v2 LoRA + long-prompt support"
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (-1 for random)")
    parser.add_argument("--width", type=int, default=WIDTH, help="Image width (multiple of 32)")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Image height (multiple of 32)")
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of inference steps")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output image path")
    return parser.parse_args()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    prompt = args.prompt
    negative_prompt = args.negative_prompt
    seed = None if args.seed == -1 else args.seed
    width, height = args.width, args.height
    guidance_scale = args.guidance_scale
    num_inference_steps = args.steps
    output_path = args.output

    print("🚀 Starting image generation with Hand v2 LoRA and long-prompt support...")
    image, used_seed = generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")
    print(f"ℹ️  Seed used: {used_seed}")

    try:
        from IPython.display import display
        display(image)
    except:
        pass