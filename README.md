# ComfyUI-FluxLayerDiffuse-NF4

https://github.com/RedAIGC/Flux-version-LayerDiffuse

![image](workflow.png)

## Download models
```bash
huggingface-cli download camenduru/FLUX.1-dev-diffusers --local-dir models/checkpoints/FLUX.1-dev

huggingface-cli download priyesh17/FLUX.1-dev_Quantized_nf4 --local-dir models/checkpoints/FLUX.1-dev_Quantized_nf4

wget -O models/vae/TransparentVAE.pth "https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/resolve/main/TransparentVAE.pth?download=true"

wget -O models/loras/layerlora.safetensors "https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/resolve/main/layerlora.safetensors?download=true"
```

## Important
Currently, only **T2I mode** is supported!