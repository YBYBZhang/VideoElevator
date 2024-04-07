from safetensors import safe_open
from .convert_from_ckpt import convert_ldm_clip_checkpoint, convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint

def add_lora_weight(pipe, lora_path, use_vae=False):
    if lora_path is None:
        return pipe
    base_model_state_dict = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys(): base_model_state_dict[key] = f.get_tensor(key)
    
    if use_vae:
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, pipe.vae.config)
        pipe.vae.load_state_dict(converted_vae_checkpoint)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, pipe.unet.config)
    pipe.unet.load_state_dict(converted_unet_checkpoint, strict=False)

    # pipe.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)

    return pipe

def add_unet_lora_weight(unet, lora_path):   
    base_model_state_dict = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys(): base_model_state_dict[key] = f.get_tensor(key)
            

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, unet.config)
    unet.load_state_dict(converted_unet_checkpoint, strict=False)

    return unet

import torch
from safetensors.torch import load_file

def load_lora_weights(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline
