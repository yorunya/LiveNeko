# LiveNeko — Livestream Video Summarization Desktop App

[English](README.md) | [简体中文](README-zh.md)

LiveNeko is a desktop application that automatically summarizes
livestream VODs (e.g. Bilibili) by combining **speech recognition**, **visual
scene detection**, and an **LLM summary engine**. 

It download videos from url or use local video as input to generate a timestamp sammurize result


## Installation

For windows, install python and ffmpeg is needed:
```bash
winget install Gyan.FFmpeg
winget install Python.Python.3.12
```
The following python libs are requires:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install torchvision transformers numpy soundfile funasr df
```
 
Download a release or build it yourself, git clone current repo then run:
```bash
git clone https://huggingface.co/FunAudioLLM/SenseVoiceSmall ./model/SenseVoiceSmall
git clone https://huggingface.co/iic/speech_fsmn_vad_zh-cn-16k-common-default ./model/fsmn-vad
git clone https://huggingface.co/iic/speech_campplus_sv_zh-cn_16k-common ./model/cam++
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe -o yt-dlp.exe
```

```bash
cd tauri-app
npm run tauri build
```

## Finetune your model
The base model is the `google/vit-base-patch16-224`, download it from hugging face use as default model, or finetune it with yourself. Tere is a example by using `sample.py` and `train.py`. you can download my simple  fine-fune model for vtuber `Ace Taffy` from release.

Also, the spk/ dir provide a voiceprint for `Ace Taffy`.
