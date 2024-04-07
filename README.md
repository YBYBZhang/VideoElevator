# VideoElevator
Official pytorch implementation of "VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models"

[![arXiv](https://img.shields.io/badge/arXiv-2403.05438-b31b1b.svg)](https://arxiv.org/abs/2403.05438)
[![Project](https://img.shields.io/badge/Project-Website-orange)](https://videoelevator.github.io/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=YBYBZhang/VideoElevator)


https://github.com/YBYBZhang/VideoElevator/assets/40799060/f850bc9c-ccf6-48b3-8011-394986aade71

**VideoElevator** aims to elevate the quality of generated videos with text-to-image diffusion models. It is *training-free* and *plug-and-play* to support cooperation of various text-to-video and text-to-image diffusion models.



## News

- [04/07/2024] We release the code of VideoElevator, including three example scripts.



## Method

<p align="center">
<img src="assets/introduction.png" width="1080px"/> 
</p>
**Top:** Taking text τ as input, conventional T2V performs both temporal and spatial modeling and accumulates low-quality contents throughout sampling chain.

**Bottom:** VideoElevator explicitly decompose each step into temporal motion refining and spatial quality elevating, where the former encapsulates T2V to enhance temporal consistency and the latter harnesses T2I to provide more faithful details, e.g., dressed in suit. Empirically, applying T2V in several timesteps is enough to ensure temporal consistency.



## Setup

### 1. Download Weights

All pre-trained weights are downloaded to `checkpoints/` directory, including the pre-trained weights of text-to-video and text-to-image diffusion models. Users can download the corresponding weights according to their needs.

1. Text-to-video diffusion models: [LaVie](https://huggingface.co/Vchitect/LaVie), [ZeroScope](), [AnimateLCM](https://huggingface.co/wangfuyun/AnimateLCM).
2. Text-to-image diffusion models: [StableDiffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [StableDiffusion v2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).
3. [Optional] LoRA from Civitai:  [RCNZ Cartoon](https://civitai.com/models/66347/rcnz-cartoon-3d), [RealisticVision](https://civitai.com/models/4201/realistic-vision-v60-b1), [Lyriel](https://civitai.com/models/22922/lyriel), [ToonYou](https://civitai.com/models/30240?modelVersionId=125771).

### 2. Requirements

```shell
conda create -n videoelevator python=3.10
conda activate videoelevator
pip install -r requirements.txt
```



## Inference

We provide three example scripts of VideoElevator in `example_scripts/` directory, and recommend to run `example_scripts/sd_animatelcm.py`. To perform improved text-to-video generation, directly run command `python example_scripts/sd_animatelcm.py`. 

Notably, all scripts can run with **less than 11 GB VRAM (e.g., 2080Ti GPU)**.

**[Optional] Hyper-parameters**

You can define the following hyper-parameters, and check their effects in **Ablation studies** of [project page](https://videoelevator.github.io/):

- **stable_steps**: the choice of timestep in temporal motion refining. 
- **stable_num**: the number of steps used in T2V denoising.

## Citation

```bibtex
@article{zhang2024videoelevator,
  title={VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models},
  author={Zhang, Yabo and Wei, Yuxiang and Lin, Xianhui and Hui, Zheng and Ren, Peiran and Xie, Xuansong and Ji, Xiangyang and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2403.05438},
  year={2024}
}
```



## Acknowledgement

This repository borrows code from [Diffusers](https://github.com/huggingface/diffusers), [LaVie](https://github.com/Vchitect/LaVie), [AnimateLCM](https://github.com/G-U-N/AnimateLCM), and [FreeInit](https://github.com/TianxingWu/FreeInit). Thanks for their contributions!

