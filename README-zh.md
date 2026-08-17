# LiveNeko — 直播视频摘要桌面应用

LiveNeko 是一款桌面应用，它结合了**语音识别**、**视觉
场景检测**和**LLM摘要引擎**，能够自动摘要直播视频（例如 Bilibili 的直播视频）。

它可以从 URL 下载视频，或使用本地视频作为输入，生成带时间戳的同步结果。

## 安装
对于 Windows 系统，需要安装 Python 和 ffmpeg：

```bash
winget install Gyan.FFmpeg 
winget install Python.Python.3.12
```

需要以下 Python 库：
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchvision transformers numpy soundfile funasr df
```

下载发布版本或自行构建，使用 git clone 当前仓库，然后运行：
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

## 微调模型

基础模型是 `google/vit-base-patch16-224`，您可以从 Hugging Face 下载并用作默认模型，或者自行进行微调。这里有一个使用 `sample.py` 和 `train.py` 的示例。您可以从 `model/` release下载我为虚拟主播 `永雏塔菲` 编写的简单 FineFune 模型。

此外，`spk/` 目录提供了 `永雏塔菲` 的声纹。
