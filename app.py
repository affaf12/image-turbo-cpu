import torch
import gradio as gr
from diffusers import DiffusionPipeline

# Load the pipeline once at startup
print("Loading Z-Image-Turbo pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.float32,   # safer for CPU
    low_cpu_mem_usage=True,      # optimize memory usage
)
pipe.to("cpu")   # force CPU mode

print("Pipeline loaded!")

def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    """Generate an image from the given prompt."""
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator("cpu").manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    return image, seed

# Example prompts
examples = [
    ["Young Chinese woman in red Hanfu, intricate embroidery..."],
    ["A majestic dragon soaring through clouds at sunset..."],
    ["Cozy coffee shop interior, warm lighting, rain on windows..."],
    ["Astronaut riding a horse on Mars, cinematic lighting..."],
    ["Portrait of a wise old wizard with a glowing crystal staff..."],
]

# Custom theme with modern aesthetics (Gradio 6)
custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    text_size="lg",
    spacing_size="md",
    radius_size="lg"
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
)

# Build the Gradio interface
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# 🎨 Z-Image-Turbo\n**Ultra-fast AI image generation** • Generate stunning images in just 8 steps")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            prompt = gr.Textbox(label="✨ Your Prompt", placeholder="Describe the image...", lines=5, max_lines=10, autofocus=True)
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    height = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Height")
                    width = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Width")
                
                num_inference_steps = gr.Slider(minimum=1, maximum=20, value=9, step=1, label="Inference Steps")
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(label="🎲 Random Seed", value=True)
                    seed = gr.Number(label="Seed", value=42, precision=0, visible=False)
                
                def toggle_seed(randomize):
                    return gr.Number(visible=not randomize)
                
                randomize_seed.change(toggle_seed, inputs=[randomize_seed], outputs=[seed])
            
            generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg", scale=1)
            
            gr.Examples(examples=examples, inputs=[prompt], label="💡 Try these prompts", examples_per_page=5)
        
        with gr.Column(scale=1, min_width=320):
            output_image = gr.Image(label="Generated Image", type="pil", format="png", show_label=False, height=600, buttons=["download", "share"])
            used_seed = gr.Number(label="🎲 Seed Used", interactive=False, container=True)
    
    generate_btn.click(fn=generate_image, inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed], outputs=[output_image, used_seed])
    prompt.submit(fn=generate_image, inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed], outputs=[output_image, used_seed])

if __name__ == "__main__":
    demo.launch(theme=custom_theme)
