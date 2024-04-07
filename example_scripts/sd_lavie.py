import os
import torch
import imageio
import json
from transformers import CLIPTextModel, CLIPTokenizer
from types import SimpleNamespace
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

import sys
sys.path.insert(0, ".")
from pipelines.videoelevator_pipeline import VideoElevatorPipeline
from pipelines.lavie_models import UNet3DConditionModel
from utils.lora import add_lora_weight

def find_model(model_name):
    """
    Finds a pre-trained model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Ema existing!')
        checkpoint = checkpoint["ema"]
    return checkpoint

def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
    if subfolder is not None:
        pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

    config_file = os.path.join(pretrained_model_path, 'config.json')
    if not os.path.isfile(config_file):
        raise RuntimeError(f"{config_file} does not exist")
    with open(config_file, "r") as f:
        config = json.load(f)
    config["_class_name"] = cls.__name__
    config["down_block_types"] = [
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D"
    ]
    config["up_block_types"] = [
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D"
    ]

    config["use_first_frame"] = False

    from diffusers.utils import WEIGHTS_NAME # diffusion_pytorch_model.bin
    

    model = cls.from_config(config)
    model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
    if not os.path.isfile(model_file):
        raise RuntimeError(f"{model_file} does not exist")
    state_dict = torch.load(model_file, map_location="cpu")
    for k, v in model.state_dict().items():
        # print(k)
        if '_temp' in k:
            state_dict.update({k: v})
        if 'attn_fcross' in k: # conpy parms of attn1 to attn_fcross
            k = k.replace('attn_fcross', 'attn1')
            state_dict.update({k: state_dict[k]})
        if 'norm_fcross' in k:
            k = k.replace('norm_fcross', 'norm1')
            state_dict.update({k: state_dict[k]})

    model.load_state_dict(state_dict)

    return model

if __name__ == "__main__":
    #### Timestep choice of Temporal Motion Refining (Total=50 steps) 
    stable_steps = [45, 35, 30, 25, 20, 15]
    #### Step number of T2V used at each timestep
    stable_num = 8
    prompt = "A bigfoot walking in the snowstorm."
    save_root = f"outputs/lavie{stable_steps}"
    os.makedirs(save_root, exist_ok=True)

    pos_prompt = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
    neg_prompt = "text, watermark, copyright, blurry, nsfw, noise, quick motion, bad quality, flicker, dirty, ugly, fast motion, quick cuts, fast editing, cuts"

    lora_path = "checkpoints/SD15_LoRA/realisticVisionV60B1_v51VAE.safetensors"
    model_id = "checkpoints/stable-diffusion-v1-5"
    unet3d_id = "checkpoints/stable-diffusion-v1-4/unet"
    text3d_id = "checkpoints/stable-diffusion-v1-4/text_encoder"
    tokenizer3d_id = "checkpoints/stable-diffusion-v1-4/tokenizer"
    video_unet_id = "checkpoints/LaVie/lavie_base.pt"

    #### Load T2I model
    zero_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    zero_pipe = add_lora_weight(zero_pipe, lora_path)
    

    #### Load T2V model
    unet3d = UNet3DConditionModel.from_pretrained_2d(unet3d_id).to(dtype=torch.float16)
    state_dict = find_model(video_unet_id)
    unet3d.load_state_dict(state_dict)
    tokenizer3d = CLIPTokenizer.from_pretrained(tokenizer3d_id)
    text_encoder_3d = CLIPTextModel.from_pretrained(text3d_id, torch_dtype=torch.float16)
    
    pipe = VideoElevatorPipeline(vae=zero_pipe.vae, text_encoder=zero_pipe.text_encoder, tokenizer=zero_pipe.tokenizer,\
        unet=zero_pipe.unet, unet3d=unet3d,tokenizer3d=tokenizer3d, scheduler=zero_pipe.scheduler, \
            text_encoder3d=text_encoder_3d, safety_checker=zero_pipe.safety_checker,\
            feature_extractor=zero_pipe.feature_extractor, requires_safety_checker=False)

    #### T2I scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    
    #### T2V scheduler
    pipe.video_scheduler = DDIMScheduler.from_pretrained("checkpoints/stable-diffusion-v1-4/", 
                                    subfolder="scheduler",
                                    beta_start=0.0001, 
                                    beta_end=0.02, 
                                    beta_schedule="linear"
                                    )
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    #### Add low-pass filter
    pipe.init_filter(video_length=16, height=512, width=512, \
        filter_params=SimpleNamespace(**{"method": "gaussian", "d_s": 1.0, "d_t": 0.5}))
        
    save_path = os.path.join(save_root, prompt.replace(" ", "_")[0:50]+f".mp4")
    output = pipe(prompt=prompt+pos_prompt, stable_steps=stable_steps, stable_num=stable_num, negative_prompt=neg_prompt, \
        num_inference_steps=50, alpha=1, video_length=16, generator=torch.Generator(42),\
            height=512,width=512,\
            )            
    result = [(r * 255).astype("uint8") for r in output.images]
    imageio.mimsave(save_path, result, fps=8)
