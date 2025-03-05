# Hunyuan_Keyframe_Lite
Cleaned up parameterized script for running dashtoon's Hunyuan Keyframe Control lora for image-to-video with start and end frames. Also slight fixes for CPU offloading to work properly (out of memory / VRAM otherwise), added flash/sage attention options from main repo, and added ffmpeg video rendering for adjustable bitrate, batch video options.

Original script: https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora

Heavier main repo with training options etc: https://github.com/dashtoon/hunyuan-video-keyframe-control-lora

The script needs at least 40GB VRAM to run, so you're looking at renting an H100 to run this on Runpod or other cloud services.

Install Instructions on Linux on Runpod or other cloud services.  You can remove the venv/deactivate lines if running locally.
```
git clone https://github.com/pftq/Hunyuan_Keyframe_Lite
cd Hunyuan_Keyframe_Lite
wget https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora/resolve/main/i2v.sft
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
apt-get update
apt-get install -y ffmpeg
deactivate
```

If you are on cloud or Runpod, rerun this each time for ffmpeg to persist and for the models to download to the volume instead of container:
```
cd Hunyuan_Keyframe_Lite
export HF_HOME=/workspace/
source venv/bin/activate
apt-get update
apt-get install -y ffmpeg
deactivate
```

Running the script. Settings are default to Dashtoon's original script. To use this on Skyreels, start prompt with "FPS-24, " and switch model_id to Skywork/SkyReels-V1-Hunyuan-I2V
```
# create a video, can switch model_id to Skywork/SkyReels-V1-Hunyuan-I2V
python video_generate.py \
--model_id "Skywork/SkyReels-V1-Hunyuan-I2V" \
--lora_path "i2v.sft" \
--cfg 6 \
--steps 50 \
--width 720 \
--height 1280 \
--image1 "https://content.dashtoon.ai/stability-images/e524013d-55d4-483a-b80a-dfc51d639158.png" \
--image2 "https://content.dashtoon.ai/stability-images/0b29c296-0a90-4b92-96b9-1ed0ae21e480.png" \
--prompt "a woman" \
--mbps 15 \
--video_num 1
```
