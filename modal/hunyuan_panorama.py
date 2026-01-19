"""
HunyuanWorld 1.0 Panorama Generation on Modal

Uses FLUX.1-dev as base model with HunyuanWorld LoRA for 360° panoramas.
Cost estimate: ~2-5 cents per image on A100 (50 inference steps @ ~30-60 sec)
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "Pillow",
        "huggingface_hub",
        "peft",  # For LoRA loading
        "fastapi",
    )
)

app = modal.App("hunyuan-panorama", image=image)

model_cache = modal.Volume.from_name("hunyuan-panorama-cache", create_if_missing=True)


@app.cls(
    gpu="A100",  # FLUX needs more VRAM; A100-40GB should work with CPU offload
    timeout=600,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class HunyuanPanorama:
    @modal.enter()
    def load_model(self):
        """Load FLUX + HunyuanWorld LoRA on container startup."""
        import os
        import torch

        os.environ["HF_HOME"] = "/cache/huggingface"

        from diffusers import FluxPipeline
        from huggingface_hub import hf_hub_download

        print("Loading FLUX.1-dev base model...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            cache_dir="/cache/huggingface",
        )

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()

        print("Loading HunyuanWorld LoRA weights...")
        # Download and load LoRA weights
        lora_path = hf_hub_download(
            repo_id="tencent/HunyuanWorld-1",
            filename="HunyuanWorld-PanoDiT-Text/lora.safetensors",
            cache_dir="/cache/huggingface",
        )
        self.pipe.load_lora_weights(lora_path)

        model_cache.commit()
        print("Model loaded and ready!")

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 960,
        num_inference_steps: int = 50,
        guidance_scale: float = 30.0,
    ) -> bytes:
        """
        Generate a 360° equirectangular panorama from a text prompt.

        Args:
            prompt: Text description of the scene
            width: Output width (default 1920, should be 2:1 ratio)
            height: Output height (default 960)
            num_inference_steps: Denoising steps (default 50, can reduce to 30 for speed)
            guidance_scale: CFG scale (default 30.0 per HunyuanWorld)

        Returns:
            PNG bytes of the generated panorama
        """
        import io
        import torch

        print(f"Generating panorama: {prompt[:100]}...")

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        # Convert to PNG bytes
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        print(f"Generated {len(png_bytes)} bytes")
        return png_bytes

    @modal.method()
    def health_check(self) -> dict:
        """Check if model is loaded and ready."""
        import torch

        return {
            "status": "healthy",
            "model_loaded": self.pipe is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else 0,
        }


@app.function(timeout=600)
@modal.fastapi_endpoint(method="POST")
def generate_panorama(item: dict) -> dict:
    """
    HTTP endpoint for panorama generation.

    POST body: {
        "prompt": "A serene mountain landscape at sunset",
        "width": 1920,
        "height": 960,
        "num_inference_steps": 50,
        "guidance_scale": 30.0
    }
    Returns: {"image_base64": "...", "width": ..., "height": ...}
    """
    import base64

    prompt = item.get("prompt", "A beautiful landscape")
    width = item.get("width", 1920)
    height = item.get("height", 960)
    steps = item.get("num_inference_steps", 50)
    guidance = item.get("guidance_scale", 30.0)

    generator = HunyuanPanorama()
    png_bytes = generator.generate.remote(prompt, width, height, steps, guidance)

    return {
        "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
        "width": width,
        "height": height,
    }


@app.local_entrypoint()
def main(
    prompt: str = "A mystical enchanted forest with ancient trees, glowing mushrooms, and fireflies at twilight",
    steps: int = 50,
):
    """Test the panorama generator."""
    generator = HunyuanPanorama()

    print("Checking health...")
    health = generator.health_check.remote()
    print(f"Health: {health}")

    print(f"Generating panorama ({steps} steps): {prompt}")
    import time

    start = time.time()
    png_bytes = generator.generate.remote(prompt, 1920, 960, steps, 30.0)
    elapsed = time.time() - start

    output_path = "test_panorama.png"
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"Saved to {output_path} ({len(png_bytes)} bytes) in {elapsed:.1f}s")
