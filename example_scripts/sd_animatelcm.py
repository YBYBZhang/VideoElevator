import os
import imageio
import torch
from types import SimpleNamespace
from diffusers import AnimateDiffPipeline, StableDiffusionPipeline, MotionAdapter,\
    DDIMScheduler, DDIMInverseScheduler, LCMScheduler

import sys
sys.path.insert(0, ".")
from pipelines.videoelevator_pipeline import VideoElevatorPipeline
from utils.lora import add_lora_weight



if __name__ == "__main__":
    #### Timestep choice of Temporal Motion Refining (Total=30 steps) 
    stable_steps = [26,21,15,9]
    #### Step number of T2V used at each timestep
    stable_num = 6
    prompt = "A dog swimming."
    save_root = f"outputs/animatelcm"
    os.makedirs(save_root, exist_ok=True)
    
    pos_prompt = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
    neg_prompt = "text, watermark, copyright, blurry, nsfw, noise, quick motion, bad quality, flicker, dirty, ugly, fast motion, quick cuts, fast editing, cuts"


    lora_path = "checkpoints/SD15_LoRA/rcnzCartoon3d_v20.safetensors"
    model_id = "checkpoints/stable-diffusion-v1-5"
    model3d_id = "checkpoints/AnimateLCM"

    adapter = MotionAdapter.from_pretrained(model3d_id, torch_dtype=torch.float16)
    zero_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    zero_pipe = add_lora_weight(zero_pipe, lora_path)
    
    vid_pipe = AnimateDiffPipeline.from_pretrained(model_id, unet=zero_pipe.unet, motion_adapter=adapter, torch_dtype=torch.float16)
    vid_pipe.load_lora_weights(model3d_id, weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    vid_pipe.set_adapters(["lcm-lora"], [0.8])

    pipe = VideoElevatorPipeline(vae=zero_pipe.vae, text_encoder=zero_pipe.text_encoder, tokenizer=zero_pipe.tokenizer,\
        unet=zero_pipe.unet, unet3d=vid_pipe.unet,tokenizer3d=vid_pipe.tokenizer, scheduler=zero_pipe.scheduler, \
            text_encoder3d=vid_pipe.text_encoder, safety_checker=zero_pipe.safety_checker,\
            feature_extractor=zero_pipe.feature_extractor, requires_safety_checker=False)

    #### Define T2I and T2V schedulers respectively
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="scaled_linear")
    pipe.video_scheduler = LCMScheduler.from_config(vid_pipe.scheduler.config, beta_schedule="linear")
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.is_animatelcm = True  # AnimateLCM only uses LCMScheduler, whose hyper-parameters are significantly different from DDIM.

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")

    #### Add low-pass filter
    pipe.init_filter(video_length=16, height=512, width=512, \
        filter_params=SimpleNamespace(**{"method": "gaussian", "d_s": 1.0, "d_t": 0.6}))
    
    save_path = os.path.join(save_root, prompt.replace(" ", "_")[0:50]+f".mp4")
    output = pipe(prompt=prompt+pos_prompt, stable_steps=stable_steps, stable_num=stable_num, negative_prompt=neg_prompt, \
        alpha=0.5, video_length=16, generator=torch.Generator(42), height=512,width=512,\
            num_inference_steps=30, video_guidance_scale=2.0, guidance_scale=7.5)            
    result = [(r * 255).astype("uint8") for r in output.images]
    imageio.mimsave(save_path, result, fps=8)