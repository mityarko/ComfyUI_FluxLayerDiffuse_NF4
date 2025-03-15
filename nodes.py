import os
import torch
import numpy as np
import comfy.model_management as mm
from PIL import Image
from diffusers import FluxPipeline
from .lib_layerdiffuse.pipeline_flux_img2img import FluxImg2ImgPipeline
from .lib_layerdiffuse.vae import TransparentVAE, pad_rgb
from huggingface_hub import hf_hub_download


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

    def load_model(self, ckpt_path, load_local_model, load_t2i, load_i2i, *args, **kwargs):
        _DTYPE = torch.bfloat16
        device = mm.get_torch_device()

        if load_local_model:
            vae_path = kwargs.get("local_vae_path", "./models/TransparentVAE.pth")
            lora_path = kwargs.get("local_lora_path", "./models/layerlora.safetensors")
            flux_path = kwargs.get("local_flux_path", "black-forest-labs/FLUX.1-dev")
        else:
            vae_path = hf_hub_download(repo_id="RedAIGC/Flux-version-LayerDiffuse", filename="TransparentVAE.pth")
            lora_path = hf_hub_download(repo_id="RedAIGC/Flux-version-LayerDiffuse", filename="layerlora.safetensors")
            flux_path = ckpt_path

        # 加载 TransparentVAE
        trans_vae = TransparentVAE(None, _DTYPE)
        trans_vae.load_state_dict(torch.load(vae_path), strict=False)
        trans_vae.to(device)

        model_dict = {"trans_vae": trans_vae}

        # 加载 T2I 模型
        if load_t2i:
            pipe_t2i = FluxPipeline.from_pretrained(ckpt_path, torch_dtype=_DTYPE).to(device)
            pipe_t2i.load_lora_weights(lora_path)
            model_dict["pipe_t2i"] = pipe_t2i

        # 加载 I2I 模型
        if load_i2i:
            pipe_i2i = FluxImg2ImgPipeline.from_pretrained(ckpt_path, torch_dtype=_DTYPE).to(device)
            pipe_i2i.load_lora_weights(lora_path)
            model_dict["pipe_i2i"] = pipe_i2i

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

        with torch.no_grad():
            _, x = trans_vae.decode(latents)

        x = x.clamp(0, 1)
        x = x.permute(0, 2, 3, 1)
        img = Image.fromarray((x * 255).float().cpu().numpy().astype(np.uint8)[0])
        return (convert_preview_image([img]),)


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

        # 将输入图像转换为适合模型的格式
        original_image = pil2tensor(tensor2pil(image)).to("cuda")
        padding_feed = [x for x in original_image.movedim(1, -1).float().cpu().numpy()]
        list_of_np_rgb_padded = [pad_rgb(x) for x in padding_feed]
        rgb_padded_bchw_01 = torch.from_numpy(np.stack(list_of_np_rgb_padded, axis=0)).float().movedim(-1, 1).to("cuda")
        original_image_feed = original_image.clone()
        original_image_feed[:, :3, :, :] = original_image_feed[:, :3, :, :] * 2.0 - 1.0
        original_image_rgb = original_image_feed[:, :3, :, :] * original_image_feed[:, 3, :, :]

        initial_latent = trans_vae.encode(original_image_feed, original_image_rgb, rgb_padded_bchw_01, use_offset=True)

        latents = pipe(
            latents=initial_latent,
            image=original_image,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cuda").manual_seed(seed),
            guidance_scale=guidance_scale,
            strength=strength,
        ).images

        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        with torch.no_grad():
            _, x = trans_vae.decode(latents)

        x = x.clamp(0, 1)
        x = x.permute(0, 2, 3, 1)
        img = Image.fromarray((x * 255).float().cpu().numpy().astype(np.uint8)[0])
        return (convert_preview_image([img]),)


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
