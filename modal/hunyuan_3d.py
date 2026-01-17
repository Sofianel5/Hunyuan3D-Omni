"""
Modal API service for Hunyuan3D-Omni - persistent web endpoint.

This creates a FastAPI web service that stays running and handles requests.
"""

import base64
import io
import os

import modal
from fastapi import HTTPException, Request

app = modal.App("hunyuan3d-omni-api")

# Default bbox is [length, height, width] in range 0~1.
DEFAULT_BBOX = (0.8, 0.64, 1.0)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "wget",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "clang",
        "libxrender1",
        "libxi6",
        "libxkbcommon0",
        "libsm6",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "ninja==1.11.1.1",
        "pybind11==2.13.4",
        "wheel",
        "setuptools",
        "transformers==4.46.0",
        "diffusers==0.30.0",
        "accelerate==1.1.1",
        "pytorch-lightning==1.9.5",
        "huggingface-hub[hf_xet]==0.30.2",
        "safetensors==0.4.4",
        "numpy==1.24.4",
        "scipy==1.14.1",
        "einops==0.8.0",
        "pandas==2.2.2",
        "opencv-python==4.10.0.84",
        "imageio==2.36.0",
        "scikit-image==0.24.0",
        "rembg==2.0.65",
        "realesrgan==0.3.0",
        "tb-nightly==2.18.0a20240726",
        "basicsr==1.4.2",
        "trimesh==4.4.7",
        "pymeshlab==2022.2.post3",
        "pygltflib==1.16.3",
        "xatlas==0.0.9",
        "open3d==0.18.0",
        "omegaconf==2.3.0",
        "pyyaml==6.0.2",
        "configargparse==1.7",
        "gradio==5.33.0",
        "fastapi==0.115.12",
        "uvicorn==0.34.3",
        "tqdm==4.66.5",
        "psutil==6.0.0",
        "cupy-cuda12x==13.4.1",
        "onnxruntime==1.16.3",
        "torchmetrics==1.6.0",
        "pydantic==2.10.6",
        "timm",
        "pythreejs",
        "torchdiffeq",
    )
    .pip_install(
        "bpy==4.0",
        extra_index_url="https://download.blender.org/pypi/",
    )
    .pip_install("deepspeed==0.17.6")
    .run_commands(
        "mkdir -p /root/ckpt",
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /root/ckpt/RealESRGAN_x4plus.pth",
    )
    .add_local_dir("hy3dshape", "/root/hy3dshape")
    .add_local_dir("hy3dpaint", "/root/hy3dpaint")
)

model_volume = modal.Volume.from_name("hunyuan3d-models", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100",
    timeout=1200,
    volumes={"/models": model_volume},
    scaledown_window=300,
    secrets=[],
)
class Hunyuan3DModel:
    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import os
        import subprocess
        import sys

        import torch

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sys.path.insert(0, "/root")

        sys.path.insert(0, "/root/hy3dpaint")
        sys.path.insert(0, "/root/hy3dpaint/custom_rasterizer")

        print(
            "Compiling custom_rasterizer with CUDA (this may take 5-10 minutes on first run)..."
        )
        subprocess.run(
            ["pip", "install", "--no-build-isolation", "-e", "."],
            cwd="/root/hy3dpaint/custom_rasterizer",
            check=True,
        )
        print("custom_rasterizer compiled!")

        print("Compiling DifferentiableRenderer...")
        subprocess.run(
            [
                "bash",
                "-c",
                "c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp -o mesh_inpaint_processor$(python3-config --extension-suffix)",
            ],
            cwd="/root/hy3dpaint/DifferentiableRenderer",
            check=True,
        )
        print("DifferentiableRenderer compiled!")

        from hy3dpaint.utils.torchvision_fix import apply_fix

        apply_fix()

        from hy3dpaint.textureGenPipeline import (
            Hunyuan3DPaintConfig,
            Hunyuan3DPaintPipeline,
        )
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

        print("Loading Hunyuan3D-Omni shape model...")
        self.shape_pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-Omni", cache_dir="/models", fast_decode=False
        )
        print("Shape model loaded successfully!")

        print("Loading Hunyuan3D-Paint texture model...")
        self.paint_pipeline = Hunyuan3DPaintPipeline(
            Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        )
        print("Paint model loaded successfully!")

    @modal.method()
    def generate(
        self, image_bytes: bytes, bbox: list[float] | None = None, add_texture: bool = True
    ) -> dict:
        """Generate 3D model from image and bbox, optionally with PBR textures."""
        import os
        import sys

        import torch
        from PIL import Image

        sys.path.insert(0, "/root")
        from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

        output_dir = "/tmp/output"
        os.makedirs(output_dir, exist_ok=True)

        temp_image_path = os.path.join(output_dir, "input_image.png")
        image = Image.open(io.BytesIO(image_bytes))
        image.save(temp_image_path)

        if bbox is None:
            bbox = DEFAULT_BBOX
        bbox_tensor = torch.FloatTensor(bbox).unsqueeze(0).unsqueeze(0)
        bbox_tensor = bbox_tensor.to(self.shape_pipeline.device).to(
            self.shape_pipeline.dtype
        )

        print("Generating 3D geometry...")
        result = self.shape_pipeline(
            image=temp_image_path,
            bbox=bbox_tensor,
            num_inference_steps=50,
            octree_resolution=512,
            mc_level=0,
            guidance_scale=4.5,
            generator=torch.Generator("cuda").manual_seed(1234),
        )

        mesh = result["shapes"][0][0]
        sampled_point = result["sampled_point"][0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)

        output_obj = os.path.join(output_dir, "output.obj")
        mesh.export(output_obj)

        if add_texture:
            print("Generating PBR textures...")
            textured_mesh_path = self.paint_pipeline(
                mesh_path=output_obj,
                image_path=temp_image_path,
                output_mesh_path=None,
                use_remesh=True,
                save_glb=True,
            )
            output_glb = textured_mesh_path.replace(".obj", ".glb")
            print(f"Textured mesh saved to {output_glb}")
        else:
            output_glb = os.path.join(output_dir, "output.glb")
            mesh.export(output_glb)

        # Read files as bytes
        with open(output_glb, "rb") as f:
            glb_bytes = f.read()

        return {
            "glb": glb_bytes,
            "success": True,
            "textured": add_texture,
        }


@app.function(
    image=image, timeout=1200, secrets=[modal.Secret.from_name("hunyuan3d-api")]
)
@modal.fastapi_endpoint(method="POST")
def generate_3d(request: dict, http_request: Request):
    """
    Web endpoint for 3D generation with optional PBR texturing.

    POST /generate_3d
    Body: {
        "image": "base64_encoded_image",
        "bbox": [x, y, z],  // optional, defaults to [0.8, 0.64, 1.0]
        "add_texture": true  // optional, default true - adds PBR materials (albedo, metallic, roughness)
    }

    Returns: {
        "glb": "base64_encoded_glb_file",
        "success": true,
        "textured": true
    }
    """
    api_key = os.environ.get("HUNYUAN3D_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server auth not configured")
    auth_header = http_request.headers.get("authorization", "")
    if auth_header != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    image_bytes = base64.b64decode(request["image"])
    bbox = request.get("bbox")
    if bbox is None:
        bbox = list(DEFAULT_BBOX)
    elif not isinstance(bbox, list) or len(bbox) != 3:
        raise HTTPException(status_code=400, detail="bbox must be [length, height, width]")
    add_texture = request.get("add_texture", True)

    model = Hunyuan3DModel()
    result = model.generate.remote(image_bytes, bbox, add_texture)

    return {
        "glb": base64.b64encode(result["glb"]).decode("utf-8"),
        "success": result["success"],
        "textured": result["textured"],
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "Hunyuan3D-Omni"}
