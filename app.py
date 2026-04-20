import os
import torch
import streamlit as st
from diffusers import DiffusionPipeline

# Hugging Face Token from Streamlit secrets
HF_TOKEN = st.secrets"HF_TOKEN = "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" "

# Load pipeline
st.write("Loading Z-Image-Turbo pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_auth_token=HF_TOKEN   # <-- Token securely injected
)
pipe.to("cpu")
st.write("Pipeline loaded!")

def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed):
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator("cpu").manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    return image, seed

# Streamlit UI
st.title("🎨 Z-Image-Turbo (CPU Mode)")
st.markdown("**Ultra-fast AI image generation • Generate stunning images in just 8 steps**")

prompt = st.text_area("✨ Your Prompt", "", height=150)
height = st.slider("Height", 512, 2048, 1024, 64)
width = st.slider("Width", 512, 2048, 1024, 64)
steps = st.slider("Inference Steps", 1, 20, 9, 1)
randomize = st.checkbox("🎲 Random Seed", True)
seed = 42

examples = [
    "Young Chinese woman in red Hanfu, intricate embroidery...",
    "A majestic dragon soaring through clouds at sunset...",
    "Cozy coffee shop interior, warm lighting, rain on windows...",
    "Astronaut riding a horse on Mars, cinematic lighting...",
    "Portrait of a wise old wizard with a glowing crystal staff...",
]
st.markdown("💡 Try these prompts")
for ex in examples:
    if st.button(ex):
        prompt = ex

if st.button("🚀 Generate Image"):
    if prompt.strip() != "":
        image, used_seed = generate_image(prompt, height, width, steps, seed, randomize)
        st.image(image, caption=f"Seed used: {used_seed}")
    else:
        st.warning("Please enter a prompt first!")
