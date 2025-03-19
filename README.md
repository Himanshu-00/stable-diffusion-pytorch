# Stable_diffusion_pytorch

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Himanshu-00/stable-diffusion-pytorch/blob/main/output/demo/demo_file.ipynb)

Yet another PyTorch implementation of [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release).

I tried my best to follow up original SD architecture and easy to read. Configs are hard-coded (based on Stable Diffusion v1.x).


Heavily referred to following repositories. Big kudos to them!

* [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
* [huggingface/diffusers/](https://github.com/huggingface/diffusers/)
* [hkproj/pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion)
* [kjsman/stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch)

## Dependencies

* PyTorch
* Numpy
* Pillow
* regex
* tqdm
* safetensors

## TODO
* SDXL Support
* InPainting

## How to Install

1. Clone or download this repository.
2. Install dependencies: Run `pip install torch numpy Pillow regex` or `pip install -r requirements.txt`.
3. Download the model from [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors) and put in the parent folder of `stable_diffusion`. Your folders should be like this:
```
stable-diffusion-pytorch(-main)/
├─ sd1_tokenizer_/
│  ├─ ...
├─ checkpoints/
│  ├─ safetensors/
│  ├─ ...
├─ stable_diffusion/
│  ├─ samplers/
└  ┴─ ...
```
*Feel free to use any SDv1.x models*

## How to Use

Import `stable_diffusion_pytorch` as submodule.

Here's some example scripts. You can also read the docstring of `stable_diffusion_pytorch.pipeline.generate`.

Text-to-image generation:
```py
python3 stable_diffusion/Inference.py \
  --prompt "A futuristic cityscape at sunset, cinematic lighting" \
  --cfg_scale 7 \
  --steps 50 \
  --seed 42
```

...with unconditional(negative) prompts:
```py
python3 stable_diffusion/Inference.py \
  --prompt "A futuristic cityscape at sunset, cinematic lighting" \
  ----uncond_prompt "(worst quality:2),(low quality:2),(normal quality:2),lowres,watermark," \
  --cfg_scale 7 \
  --steps 50 \
  --seed 42
```

Image-to-image generation:
```py
python3 stable_diffusion/Inference.py \
  --prompt "Watercolor painting of a mountain landscape" \
  --input_image input.jpg \
  --strength 0.85 \
  --sampler ddim
```

Use different sampler:
```py
python3 stable_diffusion/Inference.py \
  --prompt "Watercolor painting of a mountain landscape" \
  --input_image input.jpg \
  --strength 0.85 \
  --sampler ddim # 'DDPM', 'Euler A' and more are  avaliable 
```

## LICENSE

All codes on this repository are licensed with MIT License. Please see LICENSE file.

Note that checkpoint files of Stable Diffusion are licensed with [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) License. It has use-based restriction caluse, so you'd better read it.
