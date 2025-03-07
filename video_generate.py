from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import safetensors.torch
import torch
import torchvision.transforms.v2 as transforms
from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import HunyuanVideoLoraLoaderMixin
from diffusers.models import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoPatchEmbed, HunyuanVideoTransformer3DModel
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE, retrieve_timesteps
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, is_torch_xla_available, load_image, logging, replace_example_docstring
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from typing import Dict, List
from typing import Any
from typing import Callable
import argparse
import os
import time
import random

# 20250305 pftq load settings for customization ####
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_id", type=str, default="hunyuanvideo-community/HunyuanVideo")
parser.add_argument("--transformer_model_id", type=str, default="hunyuanvideo-community/HunyuanVideo")
parser.add_argument("--lora_path", type=str, default="i2v.sft")
parser.add_argument("--use_sage", action="store_true")
parser.add_argument("--use_flash", action="store_true")
parser.add_argument("--cfg", type=float, default=6.0)
parser.add_argument("--num_frames", type=int, default=77)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=-1) 
parser.add_argument("--prompt", type=str, default="a woman")
parser.add_argument("--height", type=int, default=1280)
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--video_num", type=int, default=1)
parser.add_argument("--image1", type=str, default="https://content.dashtoon.ai/stability-images/e524013d-55d4-483a-b80a-dfc51d639158.png")
parser.add_argument("--image2", type=str, default="https://content.dashtoon.ai/stability-images/0b29c296-0a90-4b92-96b9-1ed0ae21e480.png")
parser.add_argument("--image3", type=str, default="")
parser.add_argument("--image4", type=str, default="")
parser.add_argument("--image5", type=str, default="")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--mbps", type=float, default=7)
parser.add_argument("--color_match", action="store_true")

args = parser.parse_args()

# 20250305 pftq: from main repo at https://github.com/dashtoon/hunyuan-video-keyframe-control-lora/blob/main/hv_control_lora_inference.py
use_sage = False
use_flash = False
if args.use_sage:
    try:
        from sageattention import sageattn, sageattn_varlen
        use_sage = True
    except ImportError:
        sageattn, sageattn_varlen = None, None
elif args.use_flash:
    try:
        import flash_attn
        from flash_attn.flash_attn_interface import _flash_attn_forward, flash_attn_varlen_func
        use_flash = True
    except ImportError:
        flash_attn, _flash_attn_forward, flash_attn_varlen_func = None, None, None
print("Using SageAtten: "+str(use_sage))
print("Using FlashAttn: "+str(use_flash))


video_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]

    return image

# 20250305 pftq: from main repo at https://github.com/dashtoon/hunyuan-video-keyframe-control-lora/blob/main/hv_control_lora_inference.py
def get_cu_seqlens(attention_mask):
    """Calculate cu_seqlens_q, cu_seqlens_kv using attention_mask"""
    batch_size = attention_mask.shape[0]
    text_len = attention_mask.sum(dim=-1, dtype=torch.int)
    max_len = attention_mask.shape[-1]

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i]
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens
class HunyuanVideoFlashAttnProcessor:
    def __init__(self, use_flash_attn=True, use_sageattn=False):
        self.use_flash_attn = use_flash_attn
        self.use_sageattn = use_sageattn
        if self.use_flash_attn:
            assert flash_attn is not None, "Flash attention not available"
        if self.use_sageattn:
            assert sageattn is not None, "Sage attention not available"

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = 0

        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

            txt_seq_len = encoder_hidden_states.shape[1]

        max_seqlen_q = max_seqlen_kv = img_seq_len + txt_seq_len
        cu_seqlens_q = cu_seqlens_kv = get_cu_seqlens(attention_mask)

        query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        key = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[3])
        value = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[3])

        if self.use_flash_attn:
            hidden_states = flash_attn_varlen_func(
                query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
        elif self.use_sageattn:
            hidden_states = sageattn_varlen(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        else:
            raise NotImplementedError("Please set use_flash_attn=True or use_sageattn=True")

        hidden_states = hidden_states.reshape(batch_size, max_seqlen_q, -1)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

@torch.inference_mode()
def call_pipe(
    pipe,
    prompt: Union[str, List[str]] = None,
    prompt_2: Union[str, List[str]] = None,
    height: int = 720,
    width: int = 1280,
    num_frames: int = 129,
    num_inference_steps: int = 50,
    sigmas: List[float] = None,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
    max_sequence_length: int = 256,
    image_latents: Optional[torch.Tensor] = None,
):


    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds,
        callback_on_step_end_tensor_inputs,
        prompt_template,
    )

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_template=prompt_template,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
        max_sequence_length=max_sequence_length,
    )

    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
    )

    # 5. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_latent_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 6. Prepare guidance condition
    guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    # 20250305 pftq: added to properly offload to CPU, was out of memory otherwise
    pipe.text_encoder.to("cpu") 
    pipe.text_encoder_2.to("cpu")  
    torch.cuda.empty_cache() 

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            pipe._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = pipe.transformer(
                hidden_states=torch.cat([latent_model_input, image_latents], dim=1),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
    pipe._current_timestep = None

    if not output_type == "latent":
        latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        video = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    pipe.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return HunyuanVideoPipelineOutput(frames=video)

#20250305 pftq: customizable bitrate
# Function to check if FFmpeg is installed
import subprocess  # For FFmpeg functionality
def is_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# FFmpeg-based video saving with bitrate control
def save_video_with_ffmpeg(frames, output_path, fps, bitrate_mbps):
    frames = [np.array(frame) for frame in frames]
    height, width, _ = frames[0].shape
    bitrate = f"{bitrate_mbps}M"
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        output_path
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    stderr_output = process.stderr.read().decode()
    if process.returncode != 0:
        print(f"FFmpeg error: {stderr_output}")
    else:
        print(f"Video saved to {output_path} with FFmpeg")

# Fallback OpenCV-based video saving
def save_video_with_opencv(frames, output_path, fps, bitrate_mbps):
    frames = [np.array(frame) for frame in frames]
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Note: cv2.CAP_PROP_BITRATE is not supported, so bitrate_mbps is ignored
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        writer.write(frame)
    writer.release()
    print(f"Video saved to {output_path} with OpenCV (bitrate control unavailable)")

# Wrapper to choose between FFmpeg and OpenCV
def save_video_with_quality(frames, output_path, fps, bitrate_mbps):
    if is_ffmpeg_installed():
        save_video_with_ffmpeg(frames, output_path, fps, bitrate_mbps)
    else:
        print("FFmpeg not found. Falling back to OpenCV (bitrate not customizable).")
        save_video_with_opencv(frames, output_path, fps, bitrate_mbps)

# start executing here ###################
print("Initializing model...")
transformer_subfolder = "transformer"
if args.transformer_model_id == "Skywork/SkyReels-V1-Hunyuan-I2V":
    transformer_subfolder = "" # 20250305 pftq: Error otherwise - Skywork/SkyReels-V1-Hunyuan-I2V does not appear to have a file named config.json.
transformer = HunyuanVideoTransformer3DModel.from_pretrained(args.transformer_model_id, subfolder=transformer_subfolder, torch_dtype=torch.bfloat16)
pipe = HunyuanVideoPipeline.from_pretrained(args.base_model_id, transformer=transformer, torch_dtype=torch.bfloat16)

# Enable memory savings
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

# Apply flash attention to all transformer blocks
if use_sage or use_flash:
    for block in pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks:
        block.attn.processor = HunyuanVideoFlashAttnProcessor(use_flash_attn=use_flash, use_sageattn=use_sage)
    
with torch.no_grad():  # enable image inputs
    initial_input_channels = pipe.transformer.config.in_channels
    new_img_in = HunyuanVideoPatchEmbed(
        patch_size=(pipe.transformer.config.patch_size_t, pipe.transformer.config.patch_size, pipe.transformer.config.patch_size),
        in_chans=pipe.transformer.config.in_channels * 2,
        embed_dim=pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
    )
    new_img_in = new_img_in.to(pipe.device, dtype=pipe.dtype)
    new_img_in.proj.weight.zero_()
    new_img_in.proj.weight[:, :initial_input_channels].copy_(pipe.transformer.x_embedder.proj.weight)

    if pipe.transformer.x_embedder.proj.bias is not None:
        new_img_in.proj.bias.copy_(pipe.transformer.x_embedder.proj.bias)

    pipe.transformer.x_embedder = new_img_in

print("Loading lora...")
lora_state_dict = pipe.lora_state_dict(args.lora_path)
transformer_lora_state_dict = {f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.") and "lora" in k}
pipe.load_lora_into_transformer(transformer_lora_state_dict, transformer=pipe.transformer, adapter_name="i2v", _pipeline=pipe)
pipe.set_adapters(["i2v"], adapter_weights=[1.0])
pipe.fuse_lora(components=["transformer"], lora_scale=1.0, adapter_names=["i2v"])
pipe.unload_lora_weights()

print("Loading images...")
cond_frame1 = load_image(args.image1)
cond_frame2 = load_image(args.image2)

cond_frame1 = resize_image_to_bucket(cond_frame1, bucket_reso=(args.width, args.height))
cond_frame2 = resize_image_to_bucket(cond_frame2, bucket_reso=(args.width, args.height))

cond_video = np.zeros(shape=(args.num_frames, args.height, args.width, 3))

# 20250305 pftq: Optional 3rd-5th frame, sadly doesn't work so easily, needs more code
cond_frame3 = None
cond_frame4 = None
cond_frame5 = None

if args.image3 != "":
    cond_frame3 = load_image(args.image3)
    cond_frame3 = resize_image_to_bucket(cond_frame3, bucket_reso=(args.width, args.height))
    if args.image4 !="":
        cond_frame4 = load_image(args.image4)
        cond_frame4 = resize_image_to_bucket(cond_frame4, bucket_reso=(args.width, args.height))
        if args.image5 !="":
            cond_frame5 = load_image(args.image5)
            cond_frame5 = resize_image_to_bucket(cond_frame5, bucket_reso=(args.width, args.height))
            
if args.image5 != "" and args.image4 != "" and args.image3 !="" and args.image2 !="":
    cond_video[0] = np.array(cond_frame1)
    cond_video[args.num_frames//4] = np.array(cond_frame2)
    cond_video[(args.num_frames * 2 )//4] = np.array(cond_frame3)
    cond_video[(args.num_frames * 3 )//4] = np.array(cond_frame4)
    cond_video[args.num_frames -1] = np.array(cond_frame5)
elif args.image4 != "" and args.image3 !="" and args.image2 !="":
    cond_video[0] = np.array(cond_frame1)
    cond_video[args.num_frames//3] = np.array(cond_frame2)
    cond_video[(args.num_frames * 2 )//3] = np.array(cond_frame3)
    cond_video[args.num_frames -1] = np.array(cond_frame4)
elif args.image3 != "" and args.image2 !="":
    cond_video[0] = np.array(cond_frame1)
    cond_video[args.num_frames//2] = np.array(cond_frame2)
    cond_video[args.num_frames -1] = np.array(cond_frame3)
else:
    cond_video[0] = np.array(cond_frame1)
    cond_video[args.num_frames -1] = np.array(cond_frame2)

cond_video = torch.from_numpy(cond_video.copy()).permute(0, 3, 1, 2)
cond_video = torch.stack([video_transforms(x) for x in cond_video], dim=0).unsqueeze(0)

with torch.no_grad():
    image_or_video = cond_video.to(device="cuda", dtype=pipe.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
    cond_latents = pipe.vae.encode(image_or_video).latent_dist.sample()
    cond_latents = cond_latents * pipe.vae.config.scaling_factor
    cond_latents = cond_latents.to(dtype=pipe.dtype)

for idx in range(args.video_num): # 20250305 pftq: for loop for multiple videos per batch with varying seeds
        
    if args.seed == -1 or idx > 0: # 20250305 pftq: seed argument ignored if asking for more than one video
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    #20250223 pftq: More useful filename and higher customizable bitrate
    from datetime import datetime
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
    video_out_file = formatted_time+f"_hunyuankeyframe_{args.width}-{args.num_frames}f_cfg-{args.cfg}_steps-{args.steps}_seed-{args.seed}_{args.prompt[:40].replace('/','')}_{idx}"
    
    print("Starting video generation #"+str(idx)+" for "+video_out_file)
    video = call_pipe(
        pipe,
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        image_latents=cond_latents,
        width=args.width,
        height=args.height,
        guidance_scale=args.cfg,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]

    # 20250305 pftq: Color match with direct MKL and temporal smoothing
    if args.color_match:
        #save_video_with_quality(video, f"{video_out_file}_raw.mp4", args.fps, args.mbps)
        print("Applying color matching to video...")
        from color_matcher import ColorMatcher
        from color_matcher.io_handler import load_img_file
        from color_matcher.normalizer import Normalizer
        
        # Load the reference image (image1)
        ref_img = load_img_file(args.image1)  # Original load
        cm = ColorMatcher()
        matched_video = []
    
        for frame in video:
            frame_rgb = np.array(frame)  # Direct PIL to numpy
            matched_frame = cm.transfer(src=frame_rgb, ref=ref_img, method='mkl')
            matched_frame = Normalizer(matched_frame).uint8_norm()
            matched_video.append(matched_frame)
    
        video = matched_video
    # END OF COLOR MATCHING 
    
    print("Saving "+video_out_file)
    #export_to_video(final_video, "output.mp4", fps=24)
    save_video_with_quality(video, f"{video_out_file}.mp4", args.fps, args.mbps)
