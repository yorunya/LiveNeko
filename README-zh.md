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

## 模型

首次启动的向导（也可在**设置**页中配置）：

- **ASR**（必需）：选择本地模型目录，或在应用内从 Hugging Face / ModelScope
  下载 `SenseVoiceSmall`、`Fun-ASR-Nano`、`Paraformer-zh-streaming`；或使用
  在线 **Qwen3-ASR** API。
- **VAD**：无需配置 —— 应用内置 Silero VAD 模型，并在 CPU 上原生运行。
- **SPK**（可选）：`cam++` 模型目录，用于说话人识别；不配置则所有语句标记为
  `other`。
- **视觉模型（VideoNeko）**：你的微调 ViT 模型目录（含 `config.json` +
  `model.safetensors` + `preprocessor_config.json`）。

摘要引擎在设置中选择：OpenAI 兼容 API、本地 **Ollama** 服务或 **llama.cpp** 服务。

## 构建

```bash
cd tauri-app
npm install
npm run tauri build
```

设置始终保存在 `%APPDATA%\com.liveneko.desktop\config.json`。其余数据——`results\<标题>\`
（`summary.md`、`asr.txt`、`visual.txt`）、`funasr-models\` 和 `spk\`——保存在
**数据目录**中，该目录在首次启动向导中选择，之后可在设置中修改；修改时会把
现有数据移动到新目录。默认位置为 `%APPDATA%\com.liveneko.desktop\`。

## 微调模型

视觉基础模型是 `google/vit-base-patch16-224`；应用使用你指定的微调权重。仓库里的
两个脚本即可完成全部工作：`sample.py` 从标注好的片段中抽取训练帧，`train.py`
在这些帧上微调。

### 第 0 步 — 准备环境

- Python 3.10+，CUDA 版 PyTorch（CPU 也能跑，但很慢）：
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers opencv-python pillow tqdm
  ```
- 约 350 MB 磁盘存放数据集帧，另需几 GB 空闲显存（CPU 训练则为内存）。
- 首次运行 `train.py` 会从 Hugging Face 下载基础模型（约 350 MB）；网络受限时
  按脚本 docstring 所示加 `proxychains` 前缀。模型缓存在 `~/.cache/huggingface`，
  之后不再需要联网。

### 第 1 步 — 把标注片段放进 `data/`

每个片段是一个 `.mp4`，旁边放同名的 `.txt`：

```text
data/
├── taffy_stream_0102.mp4
├── taffy_stream_0102.txt
├── other_stream_0311.mp4
└── other_stream_0311.txt
```

`.txt` 里要么写**整段视频的一个标签**：

```text
watch anime
```

…要么写**按时间段的标签**，每行一条：

```text
[00:00:00-00:01:39] game
[00:01:41-00:03:56] live2d
[00:03:57-00:05:54] game
```

时间段为**左闭右开**（start-inclusive / end-exclusive）；未被任何段覆盖的秒数
在抽帧时跳过。时间戳支持 `hh:mm:ss`、`mm:ss` 或纯秒数。标签集合由标注文件推导
——想新增类别（例如 `sing`），加一段带该标签的片段即可。两种格式可在不同片段间
混用。

### 第 2 步 — 抽帧：`python sample.py`

```bash
python sample.py
```

脚本按 1 fps 抽帧（超过 `MAX_FRAMES_PER_VIDEO` = 300 帧的片段会均匀抽稀），
并按标签写入 JPEG：

```text
dataset/
├── game/taffy_stream_0102_s000000.jpg …
└── live2d/taffy_stream_0102_s000101.jpg …
```

预期控制台输出，每个片段一行：

```text
taffy_stream_0102.mp4 (3 segments): 214 frames, 45s outside segments skipped
other_stream_0311.mp4 [game]: 292 frames
```

"skipped 秒数"是正常现象：它们是没有任何段覆盖的时长。片段缺 `.txt` 会被跳过
并给出警告。参数即 `sample.py` 顶部的常量：`DATA_DIR`、`DATASET_DIR`、
`MAX_FRAMES_PER_VIDEO`（请保留上限——片段长度差距很大，不设上限会让类别失衡）。
之后想修正标注，直接移动/删除 `dataset/<tag>/` 里的帧即可，无需重新切片。

### 第 3 步 — 微调：`python train.py`

```bash
python train.py
```

脚本读取 `dataset/`（每个子文件夹一个类别；空文件夹跳过并警告），把 ViT 分类头
替换成你的标签集并开始训练：

```text
Device: cuda
Labels: {'game': 0, 'live2d': 1}
game: 512 frames
live2d: 388 frames
Epoch 1/5  train_loss=0.5123  val_acc=0.901
…
Epoch 5/5  train_loss=0.0187  val_acc=0.987
```

关注 `val_acc` 的上升——它就是质量信号。参数即 `train.py` 顶部的常量：
`EPOCHS`（5）、`BATCH_SIZE`（32）、`LR`（2e-5）、`VAL_FRACTION`（0.4，40% 帧
留作验证；数据集很小时应调低）、`SEED`（42，保证划分可复现）。GPU 上几分钟即可
完成。微调后的权重和 processor 写入 `model/`：

```text
model/config.json
model/model.safetensors
model/preprocessor_config.json
```

### 第 4 步 — 验证

1. 上面的 `model/` 目录就是应用需要的：在首次启动向导或 **设置 → VideoNeko
   模型** 中选择它（路径保存在 `config.json`）。
2. 应用内端到端检查：对任意直播录像跑一次摘要，打开数据目录中的
   `results\<标题>\visual.txt` —— 每行一条 `[hh:mm:ss-hh:mm:ss] tag`，
   与第 1 步标注语法一致：
   ```text
   [00:00:00-00:01:38] game
   [00:01:38-00:03:55] live2d
   ```

   检查标签是否与视频内容相符。
3. 重新训练：往 `data/` 加片段，重跑 `sample.py` 和 `train.py`（会覆盖
   `model/`）。注意 `sample.py` 只新增/覆盖帧——如果改过或删过标注，先删掉
   `dataset/`（或对应的旧帧），避免旧帧混入训练。只有移动了目录时才需要在应用里
   重新选择。

release 中提供了虚拟主播 `永雏塔菲` 的示例微调模型。
