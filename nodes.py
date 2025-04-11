import os
import torch
import numpy as np
import comfy.model_management as mm
import comfy.utils
from PIL import Image
from diffusers import FluxPipeline
from .lib_layerdiffuse.pipeline_flux_img2img import FluxImg2ImgPipeline
from .lib_layerdiffuse.vae import TransparentVAE, pad_rgb
from huggingface_hub import hf_hub_download
import torch.nn.functional as F


# 工具函数
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def convert_preview_image(images):
    images_tensors = []
    for img in images:
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float() / 255.
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)
    return torch.cat(images_tensors, dim=0) if len(images_tensors) > 1 else images_tensors[0]


# 模型加载节点
class FluxTransparentModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "load_t2i": ("BOOLEAN", {"default": True}),
                "load_i2i": ("BOOLEAN", {"default": False}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            }, "optional": {
                "local_flux_path": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "local_vae_path": ("STRING", {"default": "./models/TransparentVAE.pth"}),
                "local_lora_path": ("STRING", {"default": "./models/layerlora.safetensors"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "flux_transparent"

    def load_model(self, model, load_local_model, load_t2i, load_i2i, *args, **kwargs):
        _DTYPE = torch.bfloat16
        device = mm.get_torch_device()

        if load_local_model:
            vae_path = kwargs.get("local_vae_path", "./models/TransparentVAE.pth")
            lora_path = kwargs.get("local_lora_path", "./models/layerlora.safetensors")
            flux_path = kwargs.get("local_flux_path", "black-forest-labs/FLUX.1-dev")
        else:
            vae_path = hf_hub_download(repo_id="RedAIGC/Flux-version-LayerDiffuse", filename="TransparentVAE.pth")
            lora_path = hf_hub_download(repo_id="RedAIGC/Flux-version-LayerDiffuse", filename="layerlora.safetensors")
            flux_path = model

        base_vae = None
        pipe_t2i = None
        pipe_i2i = None

        # Step 1: Load at least one base FLUX pipeline to get the VAE component
        # Prioritize loading the one requested, but load T2I if neither is specified,
        # as we absolutely need the VAE.
        try:
            if load_t2i or not load_i2i: # Load T2I if requested OR if neither is requested
                print(f"Loading base FLUX T2I pipeline from: {flux_path}")
                # Potentially add variant="bf16" or fp16 if supported and desired
                pipe_t2i = FluxPipeline.from_pretrained(flux_path, torch_dtype=_DTYPE)
                base_vae = pipe_t2i.vae
                print("Base VAE extracted from T2I pipeline.")
                # Move pipe to device later, only if load_t2i is True
            elif load_i2i: # Only load I2I if specifically requested and T2I wasn't
                print(f"Loading base FLUX I2I pipeline from: {flux_path}")
                pipe_i2i = FluxImg2ImgPipeline.from_pretrained(flux_path, torch_dtype=_DTYPE)
                base_vae = pipe_i2i.vae
                print("Base VAE extracted from I2I pipeline.")
                # Move pipe to device later, only if load_i2i is True
        except Exception as e:
             print(f"Error loading base FLUX pipeline from {flux_path}: {e}")
             raise e # Re-raise the error

        if base_vae is None:
             raise ValueError("Could not load or extract the base VAE from the FLUX model. Ensure the path/name is correct.")
             print("Initializing Transparent VAE...")
        # Pass the loaded base_vae object here!
        trans_vae = TransparentVAE(sd_vae=base_vae, dtype=_DTYPE)
        print(f"Loading Transparent VAE state dict from: {vae_path}")
        state_dict = comfy.utils.load_torch_file(vae_path) # Use ComfyUI's safe loader
        trans_vae.load_state_dict(state_dict, strict=False) # Load the weights
        trans_vae.to(device) # Move the TransparentVAE components to the device
        print("Transparent VAE initialized and moved to device.")

        model_dict = {"trans_vae": trans_vae}

        if load_t2i:
            if pipe_t2i is None: # Should only happen if only I2I was loaded initially
                 print(f"Loading missing FLUX T2I pipeline from: {flux_path}")
                 pipe_t2i = FluxPipeline.from_pretrained(flux_path, torch_dtype=_DTYPE)

            print(f"Loading T2I LoRA weights from: {lora_path}")
            pipe_t2i.load_lora_weights(lora_path)
            pipe_t2i.to(device) # Move the full pipeline to the device
            model_dict["pipe_t2i"] = pipe_t2i
            print("FLUX T2I pipeline ready.")
        elif pipe_t2i is not None:
             # If T2I pipe was loaded just for the VAE but not requested, remove it to save memory
             print("Unloading T2I pipeline components (loaded only for VAE)...")
             del pipe_t2i
             pipe_t2i = None
             mm.soft_empty_cache()

        if load_i2i:
            if pipe_i2i is None: # If not loaded initially
                 print(f"Loading FLUX I2I pipeline from: {flux_path}")
                 pipe_i2i = FluxImg2ImgPipeline.from_pretrained(flux_path, torch_dtype=_DTYPE)

            print(f"Loading I2I LoRA weights from: {lora_path}")
            pipe_i2i.load_lora_weights(lora_path)
            pipe_i2i.to(device) # Move the full pipeline to the device
            model_dict["pipe_i2i"] = pipe_i2i
            print("FLUX I2I pipeline ready.")
        elif pipe_i2i is not None:
             # If I2I pipe was loaded just for the VAE but not requested, remove it
             print("Unloading I2I pipeline components (loaded only for VAE)...")
             del pipe_i2i
             pipe_i2i = None
             mm.soft_empty_cache()

        # --- FIX ENDS HERE ---

        return (model_dict,)


# T2I 节点
class FluxTransparentT2I:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "glass bottle, high quality"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 11111, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_t2i"
    CATEGORY = "flux_transparent"

    def generate_t2i(self, model, prompt, guidance_scale, num_inference_steps, width, height, seed):
        pipe = model["pipe_t2i"]
        trans_vae = model["trans_vae"]

        latents = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cuda").manual_seed(seed),
            guidance_scale=guidance_scale,
        ).images

        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        # Decode and Post-process
        with torch.no_grad():
            _, x = trans_vae.decode(latents)
        x = x.clamp(0, 1)

        x = x.permute(0, 2, 3, 1)

        if x.shape[-1] == 4:
            first_pixel = x[0, 0, 0, :]  # [C0, C1, C2, C3]
            if torch.abs(first_pixel[1] - 1.0) < 0.1 and torch.abs(first_pixel[0]) < 0.1 and torch.abs(first_pixel[2]) < 0.1:
                print("Decoder output has 4 channels, detected [R, G, B, A] order (background is green).")
            elif torch.abs(first_pixel[2] - 1.0) < 0.1 and torch.abs(first_pixel[1]) < 0.1 and torch.abs(first_pixel[3]) < 0.1:
                print("Decoder output has 4 channels, detected [A, R, G, B] order. Reordering to [R, G, B, A].")
                x = x[..., [1, 2, 3, 0]]  # Reorder: [A, R, G, B] -> [R, G, B, A]
            else:
                print("Warning: Could not determine channel order based on background color. Assuming [R, G, B, A].")
                print(f"First pixel values: {first_pixel}")

        x = x.to(dtype=torch.float32)
        print(f"Final output tensor shape: {x.shape}, dtype: {x.dtype}")
        return (x,)


# I2I 节点
class FluxTransparentI2I:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "a handsome man with curly hair, high quality"}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_i2i"
    CATEGORY = "flux_transparent"

    def generate_i2i(self, model, image, prompt, guidance_scale, strength, num_inference_steps, width, height, seed):
        pipe = model["pipe_i2i"]
        trans_vae = model["trans_vae"]
        device = mm.get_torch_device()

        # Debug: Inspect the input tensor
        print(f"Input tensor shape: {image.shape}")
        print(f"Input tensor dtype: {image.dtype}")
        print(f"Input tensor min/max: {image.min().item()}/{image.max().item()}")
        print(f"Input tensor first pixel: {image[0, 0, 0, :]}")

        # Input image preparation
        if image.shape[0] != 1:
            print(f"Warning: FluxTransparentI2I received batch size {image.shape[0]}, using only the first image.")
        pil_image = tensor2pil(image[0])

        new_width = width
        new_height = height
        if pil_image.size != (new_width, new_height):
            print(f"Resizing input image from {pil_image.size} to ({new_width}, {new_height}) to align with decoder.")
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Ensure the image is in RGBA format for consistency
        if pil_image.mode != 'RGBA':
            print("Converting PIL image to RGBA for consistent processing.")
            pil_image = pil_image.convert('RGBA')

        original_image_tensor_hwc = pil2tensor(pil_image).to(dtype=torch.float32, device=device)
        print(f"original_image_tensor_hwc shape: {original_image_tensor_hwc.shape}")

        np_image_hwc = original_image_tensor_hwc[0].cpu().numpy()
        np_rgb_padded_hwc = pad_rgb(np_image_hwc)
        print(f"pad_rgb output shape: {np_rgb_padded_hwc.shape}, expected ({new_height}, {new_width}, 3)")

        rgb_padded_bchw_01 = torch.from_numpy(np_rgb_padded_hwc).unsqueeze(0).permute(0, 3, 1, 2).float().to(device=device)
        print(f"rgb_padded_bchw_01 shape: {rgb_padded_bchw_01.shape}")

        original_image_feed = original_image_tensor_hwc.permute(0, 3, 1, 2)
        print(f"original_image_feed shape after permute: {original_image_feed.shape}")
        original_image_feed[:, :3, :, :] = original_image_feed[:, :3, :, :] * 2.0 - 1.0
        original_image_rgb = original_image_feed[:, :3, :, :] * original_image_feed[:, 3:4, :, :]
        print(f"original_image_rgb shape: {original_image_rgb.shape}")

        target_H = rgb_padded_bchw_01.shape[2]
        target_W = rgb_padded_bchw_01.shape[3]
        orig_H = original_image_feed.shape[2]
        orig_W = original_image_feed.shape[3]

        if target_H != orig_H or target_W != orig_W:
            print(f"Resizing cond_encoder inputs from ({orig_H}, {orig_W}) to ({target_H}, {target_W}) using interpolation.")
            original_image_feed_resized = F.interpolate(
                original_image_feed,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
            original_image_rgb_resized = F.interpolate(
                original_image_rgb,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
        else:
            print("Cond_encoder input dimensions align with padded input.")
            original_image_feed_resized = original_image_feed
            original_image_rgb_resized = original_image_rgb

        print(f"original_image_feed_resized shape: {original_image_feed_resized.shape}")
        print(f"original_image_rgb_resized shape: {original_image_rgb_resized.shape}")

        initial_latent = trans_vae.encode(
            original_image_feed_resized,
            original_image_rgb_resized,
            rgb_padded_bchw_01,
            use_offset=True
        )
        print(f"initial_latent shape: {initial_latent.shape}, expected: [1, 16, {new_height // 8}, {new_width // 8}]")

        # Pipeline call
        latents = pipe(
            latents=initial_latent,
            image=pil_image,
            prompt=prompt,
            height=new_height,
            width=new_width,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator(device).manual_seed(seed),
            guidance_scale=guidance_scale,
            strength=strength,
        ).images
        print(f"pipeline output latents shape: {latents.shape}")

        # Unpack and scale latents (following the original code)
        latents = pipe._unpack_latents(latents, new_height, new_width, pipe.vae_scale_factor)
        print(f"latents shape after unpack: {latents.shape}")
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        print(f"scaled latents shape: {latents.shape}")

        # Decode and Post-process
        with torch.no_grad():
            _, x = trans_vae.decode(latents)
        x = x.clamp(0, 1)

        x = x.permute(0, 2, 3, 1)

        if x.shape[-1] == 4:
            first_pixel = x[0, 0, 0, :]  # [C0, C1, C2, C3]
            if torch.abs(first_pixel[1] - 1.0) < 0.1 and torch.abs(first_pixel[0]) < 0.1 and torch.abs(first_pixel[2]) < 0.1:
                print("Decoder output has 4 channels, detected [R, G, B, A] order (background is green).")
            elif torch.abs(first_pixel[2] - 1.0) < 0.1 and torch.abs(first_pixel[1]) < 0.1 and torch.abs(first_pixel[3]) < 0.1:
                print("Decoder output has 4 channels, detected [A, R, G, B] order. Reordering to [R, G, B, A].")
                x = x[..., [1, 2, 3, 0]]  # Reorder: [A, R, G, B] -> [R, G, B, A]
            else:
                print("Warning: Could not determine channel order based on background color. Assuming [R, G, B, A].")
                print(f"First pixel values: {first_pixel}")

        x = x.to(dtype=torch.float32)
        print(f"Final output tensor shape: {x.shape}, dtype: {x.dtype}")
        return (x,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "FluxTransparentModelLoader": FluxTransparentModelLoader,
    "FluxTransparentT2I": FluxTransparentT2I,
    "FluxTransparentI2I": FluxTransparentI2I,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxTransparentModelLoader": "Flux Transparent Model Loader",
    "FluxTransparentT2I": "Flux Transparent T2I",
    "FluxTransparentI2I": "Flux Transparent I2I",
}
