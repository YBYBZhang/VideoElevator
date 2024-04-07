import torch
import imageio
import os
from diffusers.models import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from types import SimpleNamespace
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

import sys
sys.path.insert(0, ".")
from pipelines.videoelevator_pipeline import VideoElevatorPipeline
from utils.lora import add_lora_weight

if __name__ == "__main__":
    #### Timestep choice of Temporal Motion Refining (Total=50 steps) 
    stable_steps = [45,35,25,15]
    #### Step number of T2V used at each timestep
    stable_num = 8
    prompt = "Waves crashing against a lone lighthouse, ominous lighting."
    save_root = "outputs/zeroscope"
    os.makedirs(save_root, exist_ok=True)
    
    pos_prompt = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
    neg_prompt = "text, watermark, copyright, blurry, nsfw, noise, quick motion, bad quality, flicker, dirty, ugly, fast motion, quick cuts, fast editing, cuts"


    lora_path = "checkpoints/SD15_LoRA/realisticVisionV60B1_v51VAE.safetensors"
    model_id = "checkpoints/stable-diffusion-v1-5"
    unet3d_id = "checkpoints/zeroscope_v2_576w/unet"
    text3d_id = "checkpoints/zeroscope_v2_576w/text_encoder"
    tokenizer3d_id = "checkpoints/zeroscope_v2_576w/tokenizer"
    
    zero_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    zero_pipe = add_lora_weight(zero_pipe, lora_path)
    
    unet3d = UNet3DConditionModel.from_pretrained(unet3d_id, torch_dtype=torch.float16)
    unet3d.enable_forward_chunking(chunk_size=8, dim=1)
    tokenizer3d = CLIPTokenizer.from_pretrained(tokenizer3d_id)
    text_encoder_3d = CLIPTextModel.from_pretrained(text3d_id, torch_dtype=torch.float16)
    
    pipe = VideoElevatorPipeline(vae=zero_pipe.vae, text_encoder=zero_pipe.text_encoder, tokenizer=zero_pipe.tokenizer,\
        unet=zero_pipe.unet, unet3d=unet3d,tokenizer3d=tokenizer3d, scheduler=zero_pipe.scheduler, \
            text_encoder3d=text_encoder_3d, safety_checker=zero_pipe.safety_checker,\
            feature_extractor=zero_pipe.feature_extractor, requires_safety_checker=False)

    #### Load T2I model
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    #### Load T2V model
    pipe.video_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")
    
    #### Add low-pass filter
    pipe.init_filter(video_length=24, height=512, width=512, \
        filter_params=SimpleNamespace(**{"method": "gaussian", "d_s": 1.0, "d_t": 0.4}))
        
    
    save_path = os.path.join(save_root, prompt.replace(" ", "_")+f".mp4")
    output = pipe(prompt=prompt+pos_prompt, stable_steps=stable_steps, stable_num=stable_num, negative_prompt=neg_prompt, \
        num_inference_steps=50, alpha=1.5, video_length=24, generator=torch.Generator(42), height=512,width=512)            
    result = [(r * 255).astype("uint8") for r in output.images]
    imageio.mimsave(save_path, result, fps=8)
