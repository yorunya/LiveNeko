# LiveNeko — 直播视频摘要桌面应用

[English](README.md) | [简体中文](README-zh.md)

LiveNeko 是一款 Windows 桌面应用，结合**语音识别**、**视觉场景检测**和
**LLM 摘要引擎**，自动摘要直播录像（例如 Bilibili）。它可以从链接下载视频，
或使用本地视频作为输入，生成带时间戳的 Markdown 摘要。

## 环境要求

- Windows 10/11，需 WebView2（Windows 11 已内置）。
- 系统 `PATH` 中有 **Python 3.10+**，并安装支持 CUDA 的库 —— 应用调用系统
  Python，不内置 Python：
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers numpy soundfile funasr
  ```
  `huggingface_hub` / `modelscope` 仅在应用内下载模型时需要。
- `PATH` 中有 `ffmpeg`：
  ```bash
  winget install Gyan.FFmpeg
  winget install Python.Python.3.12
  ```

## 模型（用户自行提供，不随应用打包）

首次启动的向导（也可在**设置**页中配置）：

- **ASR**（必需）：选择本地模型目录，或在应用内从 Hugging Face / ModelScope
  下载 `SenseVoiceSmall`、`Fun-ASR-Nano`、`Paraformer-zh-streaming`；或使用
  在线 **Qwen3-ASR** API。
- **VAD**：无需配置 —— 应用内置 Silero VAD 模型，并在 CPU 上原生运行。
- **SPK**（可选）：`cam++` 模型目录，用于说话人识别；不配置则所有语句标记为
  `other`。
- **视觉模型（VideoNeko）**：你的微调 ViT 模型目录（含 `config.json` +
  `model.safetensors` + `preprocessor_config.json`）。

视频下载在应用内进程完成（无需 `yt-dlp.exe`）。摘要引擎在设置中选择：OpenAI
兼容 API、本地 **Ollama** 服务或 **llama.cpp** 服务。

## 构建

```bash
cd tauri-app
npm install
npm run tauri build
```

安装包输出到
`tauri-app/src-tauri/target/release/bundle/nsis/LiveNeko_<version>_x64-setup.exe`
（例如 `LiveNeko_0.1.10_x64-setup.exe`，约 9.6 MB，未签名 —— 由于未签名，首次
运行会触发 Windows SmartScreen 警告）。

设置始终保存在 `%APPDATA%\com.liveneko.desktop\config.json`。其余数据——`results\<标题>\`
（`summary.md`、`asr.txt`、`visual.txt`）、`funasr-models\` 和 `spk\`——保存在
**数据目录**中，该目录在首次启动向导中选择，之后可在设置中修改；修改时会把
现有数据移动到新目录。默认位置为 `%APPDATA%\com.liveneko.desktop\`。

## 微调模型

视觉基础模型是 `google/vit-base-patch16-224`；应用使用你指定的微调权重。
`sample.py` 从 `data/` 按 1 fps 抽帧到 `dataset/<tag>/`，`train.py` 在
`dataset/` 上微调并输出到 `model/` —— 在应用中将“视觉模型”指向该目录即可。
release 中提供了虚拟主播 `永雏塔菲` 的示例微调模型；`spk/` 中提供了她的声纹
（供独立的 `AudioNeko.py` 流水线使用，应用不使用该目录）。
