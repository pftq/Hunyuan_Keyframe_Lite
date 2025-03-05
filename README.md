# Hunyuan Keyframe Lite
Cleaned up parameterized script for running dashtoon's Hunyuan Keyframe Control lora for image-to-video with start and end frames. It also has slight fixes for CPU offloading to work properly (out of memory / VRAM otherwise even on H100), cleaned up flash/sage attention support from main repo, and quality of life improvements like ffmpeg video rendering for adjustable bitrate, batch video options, etc.  I was personally having a lot of issues getting the lora to work, so I'm just sharing here for anyone else that wants something easy to pull and start using.

Credits to Dashtoon of course for the lora & original script, this is just a "repackaging" to make it easier to use:
- Original script: https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora
- Heavier main repo with training options etc: https://github.com/dashtoon/hunyuan-video-keyframe-control-lora

The script needs at least 40GB VRAM to run, so you're looking at renting an H100/A-series or other GPU online to run this. Performance numbers on the default/example image & settings:
- H100 without Sage/Flash: 30 min
- H100 with Sage: 10 min

Install Instructions on Linux on Runpod or other cloud services.  You can remove the venv/deactivate/export lines if running locally.
```
git clone https://github.com/pftq/Hunyuan_Keyframe_Lite
cd Hunyuan_Keyframe_Lite
wget https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora/resolve/main/i2v.sft
python -m venv venv
source venv/bin/activate
export HF_HOME=/workspace/
pip install -r requirements.txt
apt-get update
apt-get install -y ffmpeg
```

If you are on cloud or Runpod, rerun this each time you start a pod/instance for ffmpeg to persist and for the models to download to the volume instead of container:
```
cd Hunyuan_Keyframe_Lite
export HF_HOME=/workspace/
source venv/bin/activate
apt-get update
apt-get install -y ffmpeg
```

Running the script. Settings are default to Dashtoon's original script. If sage and flash are both enabled, it'll prioritize using sage. Leave out the seed line for random. Can potentially stack on Skyreel by prompting with "FPS-24, " and switching transformer_model_id to Skywork/SkyReels-V1-Hunyuan-I2V but needs more code to incorporate the negative prompt / image latents first.
```
python video_generate.py \
--base_model_id "hunyuanvideo-community/HunyuanVideo" \
--transformer_model_id "hunyuanvideo-community/HunyuanVideo" \
--lora_path "i2v.sft" \
--use_sage \
--use_flash \
--cfg 6 \
--steps 50 \
--num_frames 77 \
--width 720 \
--height 1280 \
--seed 0 \
--image1 "https://content.dashtoon.ai/stability-images/e524013d-55d4-483a-b80a-dfc51d639158.png" \
--image2 "https://content.dashtoon.ai/stability-images/0b29c296-0a90-4b92-96b9-1ed0ae21e480.png" \
--prompt "a woman" \
--mbps 15 \
--video_num 1
```
