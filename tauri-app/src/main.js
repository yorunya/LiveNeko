import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { marked } from "marked";

// ---------- OS theme ----------
// The Rust backend detects the OS theme and accent color once at startup
// (official Tauri window API + official platform methods); we fetch them once
// here and apply them. `data-theme` selects color-scheme (and the scheme-tuned
// tokens in styles.css), the accent is injected as --os-accent, and readable
// text on it as --os-on-accent — every other color is derived in CSS from
// system color keywords. The app always follows the OS theme: no custom theme
// option, and no runtime theme-change listener is registered.
async function applyOsTheme() {
  let info = null;
  try {
    info = await invoke("get_os_theme");
  } catch { /* keep the system-keyword fallbacks from :root */ }
  const theme = info?.theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme; // native scrollbars/controls
  if (info?.accent) {
    const root = document.documentElement;
    root.style.setProperty("--os-accent", info.accent);
    const onAccent = contrastColor(info.accent, theme);
    if (onAccent) root.style.setProperty("--os-on-accent", onAccent);
  }
}

// Resolve a CSS system color keyword (Canvas, CanvasText, ...) to its rgb value.
function systemColor(keyword) {
  const probe = document.createElement("span");
  probe.style.color = keyword;
  document.body.appendChild(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color;
}

// Readable text on top of the OS accent: pick the light or dark pole of the
// OS theme's own canvas colors according to the accent's relative luminance.
// (Canvas is the light pole in a light scheme and the dark pole in a dark one;
// CanvasText is its opposite, so no fixed contrast colors are needed.)
function contrastColor(hex, theme) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  if ([r, g, b].some(Number.isNaN)) return null;
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  const darkPole = theme === "light" ? "CanvasText" : "Canvas";
  const lightPole = theme === "light" ? "Canvas" : "CanvasText";
  return luminance > 0.55 ? systemColor(darkPole) : systemColor(lightPole);
}

// ---------- global state ----------
const state = {
  config: null,
  env: null,
  queue: [],
  running: false,
  results: [],
  activeResult: null,
  // last FunASR model validation report (from validate_model_config)
  modelReport: null,
  // freshly picked (not yet saved) speaker reference WAV source path
  speakerWav: "",
};

// ---------- i18n ----------
const I18N = {
  en: {
    "tab.pipeline": "Pipeline",
    "tab.results": "Results",
    "tab.settings": "Settings",
    "btn.start": "Start",
    "btn.stop": "Stop",
    "stage.videoInput": "Video Input",
    "stage.asr": "ASR",
    "stage.visual": "Visual",
    "stage.summary": "Summary",
    "pipeline.addTitle": "Add videos to the queue",
    "pipeline.urlPlaceholder": "Bilibili video URL (e.g. https://www.bilibili.com/video/BV…)",
    "pipeline.addUrl": "Add URL",
    "pipeline.addLocal": "Add local videos",
    "pipeline.queue": "Queue",
    "pipeline.clear": "Clear",
    "pipeline.queueEmpty": "Queue is empty. Add a URL or local video above.",
    "pipeline.log": "Log",
    "results.searchPlaceholder": "Search titles & contents (regex supported)",
    "results.regex": "regex",
    "results.search": "Search",
    "results.clear": "Clear",
    "results.summaries": "Summaries",
    "results.selectHint": "Select a summary on the left.",
    "results.noSummaries": "No summaries yet.",
    "results.noMatch": "No matching summaries.",
    "results.deleted": "Result deleted.",
    "results.loadFailed": "Failed to load:",
    "results.resummarize": "Re-summarize",
    "results.showAsr": "Show ASR",
    "results.showVisual": "Show Visual",
    "results.copy": "Copy",
    "results.copied": "Copied ✓",
    "results.export": "Export…",
    "results.delete": "Delete",
    "results.resummarizing": "Re-summarizing with current engine…",
    "results.failed": "Failed:",
    "results.done": "Done ✓",
    "results.thinking": "🧠 Thinking",
    "results.noSummary": "*no summary*",
    "results.noAsr": "(no asr)",
    "results.noVisual": "(no visual)",
    "results.deleteConfirm": "Delete the summary, ASR and visual results for \"{stem}\"?",
    "results.exported": "Exported to",
    "results.exportFailed": "Export failed:",
    "results.deleteFailed": "Delete failed:",
    "results.matchCount": "{n} match(es)",
    "results.searchError": "Search error:",
    "settings.languageTitle": "Language",
    "settings.languageHint": "Interface language. Default follows your system language.",
    "settings.languageSystem": "System default",
    "settings.envTitle": "Environment Check",
    "settings.recheck": "Re-check environment",
    "settings.funasrTitle": "FunASR audio models",
    "settings.funasrHint": "FunASR models are not bundled with the app. Pick a local model directory, or download one from Hugging Face / ModelScope. ASR and VAD models are required; the SPK model is optional.",
    "settings.asrTitle": "ASR model",
    "settings.vadTitle": "VAD model (fsmn-vad)",
    "settings.spkModelTitle": "SPK model (cam++, optional)",
    "settings.spkModelEnable": "Use a SPK model for speaker identification",
    "settings.modelType": "Model type",
    "settings.modelSource": "Source",
    "settings.sourceLocal": "Local directory",
    "settings.modelIdPlaceholder": "Repo id (e.g. FunAudioLLM/SenseVoiceSmall)",
    "settings.asrDirPlaceholder": "Path to the ASR model directory",
    "settings.vadDirPlaceholder": "Path to the fsmn-vad model directory",
    "settings.spkDirPlaceholder": "Path to the cam++ model directory",
    "settings.languageOptional": "Language hint",
    "settings.languagePlaceholder": "optional (zh / en / auto)",
    "settings.download": "Download",
    "settings.validate": "Validate",
    "settings.valid": "valid ✓",
    "settings.invalid": "invalid",
    "settings.skipped": "not configured (optional)",
    "settings.validating": "Validating…",
    "settings.downloading": "Downloading…",
    "settings.downloadDone": "Downloaded ✓",
    "settings.downloadFailed": "Download failed:",
    "settings.downloadNeedsHub": "Choose Hugging Face or ModelScope as the source first.",
    "settings.modelsOk": "FunASR models OK ✓",
    "settings.spkDisabledHint": "Speaker identification is disabled until a SPK model is configured.",
    "env.downloadLibs": "Model download tools",
    "env.downloadLibsMissing": "missing — model downloads unavailable",
    "settings.videonekoTitle": "Visual Model",
    "settings.videonekoHint": "Visual model directory (must contain config.json, model.safetensors, preprocessor_config.json).",
    "settings.videonekoPlaceholder": "Path to VideoNeko model directory",
    "settings.browse": "Browse…",
    "settings.speakerTitle": "Speaker identification",
    "settings.speakerHint": "Optionally tag transcript lines with a specific speaker. Requires a cam++ SPK model above; choose a short reference WAV clip of that speaker's voice (a few seconds of clean speech); it is converted to 16 kHz automatically if needed.",
    "settings.speakerEnable": "Identify a specific speaker",
    "settings.speakerName": "Speaker name",
    "settings.speakerNamePlaceholder": "e.g. taffy",
    "settings.speakerBrowse": "Choose reference WAV…",
    "settings.speakerFileCurrent": "Imported reference:",
    "settings.speakerFileNew": "(new — saved with settings)",
    "settings.speakerFileNone": "No reference WAV chosen.",
    "settings.speakerNameRequired": "Enter a speaker name first.",
    "settings.speakerWavRequired": "Choose a reference WAV file for this speaker.",
    "settings.qualityTitle": "Video download quality",
    "settings.qualityHint": "Resolution used when downloading a video from a Bilibili URL (360P / 480P / 720P / 1080P). Lower is faster and smaller; default is 720P.",
    "settings.cookiesTitle": "Browser cookies",
    "settings.cookiesHint": "Optionally reuse the login cookies of a local browser for video downloads (same source as yt-dlp --cookies-from-browser). Useful for logged-in / higher-quality downloads. Downloads also work without it. Close the selected browser while downloading so its cookie database can be read.",
    "settings.cookiesBrowser": "Read cookies from",
    "settings.cookiesOff": "Disabled",
    "settings.engineTitle": "Summarization Engine",
    "settings.engineHint": "Choose where the summary LLM runs: a hosted OpenAI-compatible API, a local Ollama server, or a llama.cpp server.",
    "settings.engineApi": "OpenAI-compatible API",
    "settings.engineOllama": "Ollama",
    "settings.engineLlamacpp": "llama.cpp server",
    "settings.baseUrl": "Base URL",
    "settings.apiKey": "API Key",
    "settings.model": "Model",
    "settings.thinking": "Thinking",
    "settings.maxTokens": "Max tokens",
    "settings.temperature": "Temperature",
    "settings.topP": "Top-p",
    "settings.testConnection": "Test connection",
    "settings.promptTitle": "Summary prompt",
    "settings.promptHint": "System prompt sent to the summarization model. Edit freely; it is stored locally. Empty → reset to the bundled default.",
    "settings.resetPrompt": "Reset to default",
    "settings.save": "Save",
    "settings.saved": "Saved ✓",
    "settings.saveError": "Error:",
    "settings.promptReset": "Prompt reset to default ✓",
    "settings.resetFailed": "Reset failed:",
    "settings.testing": "Testing…",
    "settings.connectionOk": "Connection OK ✓",
    "settings.connectionOkProbe": "Credentials verified ✓ — the model rejected the silent probe clip (some models need ~2s of real speech).",
    "settings.connectionFailed": "Failed",
    "status.ready": "Ready.",
    "status.checkingEnv": "Checking environment…",
    "status.envComplete": "Environment check complete.",
    "status.envFailed": "Environment check failed:",
    "status.running": "Pipeline running…",
    "status.finished": "Pipeline finished.",
    "status.stopRequested": "Stop requested…",
    "status.addUrlFailed": "Add URL failed:",
    "status.addFileFailed": "Add file failed:",
    "status.startFailed": "Start failed:",
    "status.setupFailed": "Setup failed:",
    "status.setupComplete": "Setup complete.",
    "env.python": "Python",
    "env.ffmpeg": "ffmpeg",
    "env.cuda": "CUDA (GPU)",
    "env.pythonLibs": "Python libraries",
    "env.ok": "OK",
    "env.missing": "missing",
    "env.missingLibs": "missing:",
    "env.runCheck": "Run a check to see environment status.",
    "env.cudaOn": "on",
    "env.cudaOff": "off",
    "env.libsOk": "ok",
    "env.libsMissing": "missing",
    "env.cudaLabel": "CUDA",
    "env.libsLabel": "Python libs",
    "log.finished": "finished OK",
    "log.cancelled": "cancelled",
    "log.failed": "failed:",
    "setup.welcome": "Welcome to LiveNeko 👋",
    "setup.hint": "Let's get everything configured. Complete the steps below — the pipeline needs FunASR audio models (ASR/VAD), your VideoNeko model and a summarization engine.",
    "setup.step1": "Select your VideoNeko model weights",
    "setup.step1Hint": "Browse for the directory containing your fine-tuned ViT weights (config.json, model.safetensors, preprocessor_config.json).",
    "setup.step2": "Choose a summarization engine",
    "setup.step2Hint": "Select where the summary LLM runs: a hosted OpenAI-compatible API, a local Ollama server, or a llama.cpp server.",
    "setup.videonekoPlaceholder": "Path to VideoNeko model directory",
    "setup.engineApi": "OpenAI API",
    "setup.engineOllama": "Ollama",
    "setup.engineLlamacpp": "llama.cpp",
    "setup.apiBase": "API Base URL (e.g. https://api.openai.com/v1)",
    "setup.apiKey": "API Key (optional)",
    "setup.apiModel": "API Model name (e.g. gpt-4o)",
    "setup.ollamaBase": "Ollama Base URL (e.g. http://localhost:11434/v1)",
    "setup.ollamaModel": "Ollama model (e.g. qwen3:8b)",
    "setup.llamacppBase": "llama.cpp Base URL (e.g. http://localhost:8080/v1)",
    "setup.llamacppModel": "Model name or GGUF path",
    "setup.skip": "Skip for now",
    "setup.finish": "Finish Setup",
    "setup.stepFunasr": "Configure FunASR audio models",
    "setup.stepFunasrHint": "ASR and VAD models are not bundled. Choose a model type and download it from Hugging Face / ModelScope, or pick an existing local directory.",
    "setup.downloadVad": "Download fsmn-vad",
    "setup.funasrIncomplete": "FunASR models are not ready:",
  },
  zh: {
    "tab.pipeline": "分析",
    "tab.results": "结果",
    "tab.settings": "设置",
    "btn.start": "开始",
    "btn.stop": "停止",
    "stage.videoInput": "视频输入",
    "stage.asr": "ASR",
    "stage.visual": "视觉",
    "stage.summary": "摘要",
    "pipeline.addTitle": "添加视频到队列",
    "pipeline.urlPlaceholder": "Bilibili 视频链接（如 https://www.bilibili.com/video/BV…）",
    "pipeline.addUrl": "添加链接",
    "pipeline.addLocal": "添加本地视频",
    "pipeline.queue": "队列",
    "pipeline.clear": "清空",
    "pipeline.queueEmpty": "队列为空。请在上方添加链接或本地视频。",
    "pipeline.log": "日志",
    "results.searchPlaceholder": "搜索标题和内容（支持正则）",
    "results.regex": "正则",
    "results.search": "搜索",
    "results.clear": "清除",
    "results.summaries": "摘要",
    "results.selectHint": "在左侧选择一个摘要。",
    "results.noSummaries": "暂无摘要。",
    "results.noMatch": "没有匹配的摘要。",
    "results.deleted": "结果已删除。",
    "results.loadFailed": "加载失败：",
    "results.resummarize": "重新摘要",
    "results.showAsr": "显示 ASR",
    "results.showVisual": "显示视觉",
    "results.copy": "复制",
    "results.copied": "已复制 ✓",
    "results.export": "导出…",
    "results.delete": "删除",
    "results.resummarizing": "正在用当前引擎重新摘要",
    "results.failed": "失败：",
    "results.done": "完成 ✓",
    "results.thinking": "🧠 思考过程",
    "results.noSummary": "*无摘要*",
    "results.noAsr": "（无 ASR）",
    "results.noVisual": "（无视觉）",
    "results.deleteConfirm": "确定删除 \"{stem}\" 的摘要、ASR 和视觉结果？",
    "results.exported": "已导出到",
    "results.exportFailed": "导出失败：",
    "results.deleteFailed": "删除失败：",
    "results.matchCount": "{n} 个匹配",
    "results.searchError": "搜索错误：",
    "settings.languageTitle": "语言",
    "settings.languageHint": "界面语言。默认跟随系统语言。",
    "settings.languageSystem": "跟随系统",
    "settings.envTitle": "环境检查",
    "settings.recheck": "重新检查环境",
    "settings.funasrTitle": "FunASR 语音模型",
    "settings.funasrHint": "应用不再内置 FunASR 模型。请选择本地模型目录，或从 Hugging Face / ModelScope 下载。ASR 与 VAD 模型为必填，SPK 模型可选。",
    "settings.asrTitle": "ASR 模型",
    "settings.vadTitle": "VAD 模型（fsmn-vad）",
    "settings.spkModelTitle": "SPK 模型（cam++，可选）",
    "settings.spkModelEnable": "启用 SPK 模型进行说话人识别",
    "settings.modelType": "模型类型",
    "settings.modelSource": "来源",
    "settings.sourceLocal": "本地目录",
    "settings.modelIdPlaceholder": "仓库 ID（如 FunAudioLLM/SenseVoiceSmall）",
    "settings.asrDirPlaceholder": "ASR 模型目录路径",
    "settings.vadDirPlaceholder": "fsmn-vad 模型目录路径",
    "settings.spkDirPlaceholder": "cam++ 模型目录路径",
    "settings.languageOptional": "语言提示",
    "settings.languagePlaceholder": "可选（zh / en / auto）",
    "settings.download": "下载",
    "settings.validate": "校验",
    "settings.valid": "有效 ✓",
    "settings.invalid": "无效",
    "settings.skipped": "未配置（可选）",
    "settings.validating": "校验中…",
    "settings.downloading": "下载中…",
    "settings.downloadDone": "下载完成 ✓",
    "settings.downloadFailed": "下载失败：",
    "settings.downloadNeedsHub": "请先将来源切换为 Hugging Face 或 ModelScope。",
    "settings.modelsOk": "FunASR 模型正常 ✓",
    "settings.spkDisabledHint": "未配置 SPK 模型时将禁用说话人识别。",
    "env.downloadLibs": "模型下载工具",
    "env.downloadLibsMissing": "缺失 — 无法下载模型",
    "settings.videonekoTitle": "视觉模型",
    "settings.videonekoHint": "视觉模型目录（需包含 config.json、model.safetensors、preprocessor_config.json）。",
    "settings.videonekoPlaceholder": "视觉模型目录路径",
    "settings.browse": "浏览…",
    "settings.speakerTitle": "说话人识别",
    "settings.speakerHint": "可选：在转写结果中用指定说话人标注语句（需先在上方配置 cam++ SPK 模型）。请提供该说话人声音的参考 WAV 片段（几秒清晰语音即可）；如不是 16 kHz 会自动转换。",
    "settings.speakerEnable": "识别指定说话人",
    "settings.speakerName": "说话人名称",
    "settings.speakerNamePlaceholder": "例如 taffy",
    "settings.speakerBrowse": "选择参考 WAV…",
    "settings.speakerFileCurrent": "已导入参考：",
    "settings.speakerFileNew": "（新选择，保存设置时导入）",
    "settings.speakerFileNone": "尚未选择参考 WAV。",
    "settings.speakerNameRequired": "请先填写说话人名称。",
    "settings.speakerWavRequired": "请为该说话人选择参考 WAV 文件。",
    "settings.qualityTitle": "视频下载清晰度",
    "settings.qualityHint": "从链接下载视频时使用的分辨率（360P / 480P / 720P / 1080P）",
    "settings.cookiesTitle": "浏览器 Cookie",
    "settings.cookiesHint": "可选择在下载视频时复用本地浏览器的登录 Cookie（数据来源与 yt-dlp --cookies-from-browser 相同）。适合需要登录或更高清晰度的下载；不开启也能正常下载。下载时请先关闭所选浏览器，否则无法读取其 Cookie 数据库。",
    "settings.cookiesBrowser": "读取 Cookie 的浏览器",
    "settings.cookiesOff": "不启用",
    "settings.engineTitle": "LLM引擎",
    "settings.engineHint": "选择总结摘要的LLM：OpenAI 兼容 API或自定义本地 服务",
    "settings.engineApi": "OpenAI 兼容 API",
    "settings.engineOllama": "Ollama",
    "settings.engineLlamacpp": "llama.cpp 服务器",
    "settings.baseUrl": "基础 URL",
    "settings.apiKey": "API 密钥",
    "settings.model": "模型",
    "settings.thinking": "思考",
    "settings.maxTokens": "Max Token",
    "settings.temperature": "Temperature",
    "settings.topP": "Top-p",
    "settings.testConnection": "测试连接",
    "settings.promptTitle": "摘要提示词",
    "settings.promptHint": "发送给摘要模型的系统提示词，留空则重置为内置默认值",
    "settings.resetPrompt": "重置为默认",
    "settings.save": "保存",
    "settings.saved": "已保存 ✓",
    "settings.saveError": "错误：",
    "settings.promptReset": "提示词已重置为默认 ✓",
    "settings.resetFailed": "重置失败：",
    "settings.testing": "测试中…",
    "settings.connectionOk": "连接正常 ✓",
    "settings.connectionOkProbe": "凭证已验证 ✓ — 模型拒绝了静音测试片段（部分模型需要约 2 秒真实语音）。",
    "settings.connectionFailed": "失败",
    "status.ready": "就绪。",
    "status.checkingEnv": "正在检查环境…",
    "status.envComplete": "环境检查完成。",
    "status.envFailed": "环境检查失败：",
    "status.running": "分析中",
    "status.finished": "分析完成",
    "status.stopRequested": "已请求停止",
    "status.addUrlFailed": "添加链接失败：",
    "status.addFileFailed": "添加文件失败：",
    "status.startFailed": "启动失败：",
    "status.setupFailed": "设置失败：",
    "status.setupComplete": "设置完成。",
    "env.python": "Python",
    "env.ffmpeg": "ffmpeg",
    "env.cuda": "CUDA（GPU）",
    "env.pythonLibs": "Python 库",
    "env.ok": "正常",
    "env.missing": "缺失",
    "env.missingLibs": "缺失：",
    "env.runCheck": "运行检查以查看环境状态。",
    "env.cudaOn": "开",
    "env.cudaOff": "关",
    "env.libsOk": "正常",
    "env.libsMissing": "缺失",
    "env.cudaLabel": "CUDA",
    "env.libsLabel": "Python 库",
    "log.finished": "完成",
    "log.cancelled": "已取消",
    "log.failed": "失败：",
    "setup.welcome": "欢迎使用 LiveNeko 👋",
    "setup.hint": "让我们完成配置。完成以下步骤——分析需要提供 FunASR 语音模型（ASR/VAD）、视觉模型和摘要引擎。",
    "setup.step1": "选择视觉模型",
    "setup.step1Hint": "选择视觉模型目录（包括config.json、model.safetensors、preprocessor_config.json）。",
    "setup.step2": "选择摘要引擎",
    "setup.step2Hint": "选择摘要 LLM 的运行位置：托管的 OpenAI 兼容 API、本地 Ollama 服务器或 llama.cpp 服务器。",
    "setup.videonekoPlaceholder": "VideoNeko 模型目录路径",
    "setup.engineApi": "OpenAI API",
    "setup.engineOllama": "Ollama",
    "setup.engineLlamacpp": "llama.cpp",
    "setup.apiBase": "API Base URL（如 https://api.openai.com/v1）",
    "setup.apiKey": "API 密钥（可选）",
    "setup.apiModel": "API 模型名（如 gpt-4o）",
    "setup.ollamaBase": "Ollama 基础 URL（如 http://localhost:11434/v1）",
    "setup.ollamaModel": "Ollama 模型（如 qwen3:8b）",
    "setup.llamacppBase": "llama.cpp 基础 URL（如 http://localhost:8080/v1）",
    "setup.llamacppModel": "模型名或 GGUF 路径",
    "setup.skip": "暂时跳过",
    "setup.finish": "完成设置",
    "setup.stepFunasr": "配置 FunASR 语音模型",
    "setup.stepFunasrHint": "应用不内置 ASR/VAD 模型。请选择模型类型并从 Hugging Face / ModelScope 下载，或选择本地目录。",
    "setup.downloadVad": "下载 fsmn-vad",
    "setup.funasrIncomplete": "FunASR 模型未就绪：",
  },
};

function systemLanguage() {
  const l = (navigator.language || "en").toLowerCase();
  return l.startsWith("zh") ? "zh" : "en";
}

let lang = systemLanguage();

function t(key, params) {
  let s = I18N[lang]?.[key] ?? I18N.en[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      s = s.split(`{${k}}`).join(String(v));
    }
  }
  return s;
}

function setLanguage(l) {
  lang = l === "zh" || l === "en" ? l : systemLanguage();
  applyLanguage();
}

// ---------- helpers ----------
const $ = (sel) => document.querySelector(sel);

function setStatus(text) {
  $("#status-text").textContent = text;
}

function setModelStatus(text) {
  $("#model-status").textContent = text;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtDate(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString();
}

function fmtSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${units[i]}`;
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const name = btn.dataset.tab;
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    $(`#tab-${name}`).classList.add("active");
    if (name === "results") refreshResults();
  });
});

// ---------- queue rendering ----------
function renderQueue() {
  const list = $("#queue-list");
  if (!state.queue.length) {
    list.innerHTML = `<div class="empty">${t("pipeline.queueEmpty")}</div>`;
    return;
  }
  const stageNames = [t("stage.videoInput"), t("stage.asr"), t("stage.visual"), t("stage.summary")];
  list.innerHTML = state.queue
    .map((item) => {
      const st = item.status;
      const stageBtns = stageNames.map((name, i) => {
        const stage = i + 1;
        const pct = (item.stageProgress && item.stageProgress[i]) || 0;
        let cls = "stage-btn";
        if (pct >= 100) cls += " done";
        if (item.status === "Error") cls += " error";
        const pctLabel = pct > 0 ? `${pct}%` : "";
        return `<div class="${cls}" title="${name}">
          <span class="stage-name">${name}</span>
          <span class="stage-pct">${pctLabel}</span>
        </div>`;
      }).join("");
      return `<div class="queue-item">
        <div class="q-head">
          <span class="q-title">${esc(item.title)}</span>
          <div class="q-actions">
            <span class="status-badge status-${st}">${st}</span>
            <button class="btn ghost small" data-remove="${item.id}">✕</button>
          </div>
        </div>
        <div class="stage-row">${stageBtns}</div>
        ${item.error ? `<div class="q-error">${esc(item.error)}</div>` : ""}
      </div>`;
    })
    .join("");
  list.querySelectorAll("[data-remove]").forEach((b) => {
    b.addEventListener("click", async () => {
      await invoke("remove_item", { id: b.dataset.remove });
      refreshQueue();
    });
  });
}

async function refreshQueue() {
  state.queue = await invoke("get_queue");
  state.running = await invoke("is_running");
  renderQueue();
  renderRunControls();
}

let queueRenderQueued = false;
function scheduleQueueRender() {
  if (queueRenderQueued) return;
  queueRenderQueued = true;
  requestAnimationFrame(() => {
    queueRenderQueued = false;
    renderQueue();
  });
}

function renderRunControls() {
  const btn = $("#btn-run");
  if (state.running) {
    btn.textContent = t("btn.stop");
    btn.classList.remove("success");
    btn.classList.add("danger");
  } else {
    btn.textContent = t("btn.start");
    btn.classList.remove("danger");
    btn.classList.add("success");
  }
}

// ---------- log ----------
const logBuffer = [];
let logFlushQueued = false;

function appendLog(line) {
  logBuffer.push(line);
  if (logFlushQueued) return;
  logFlushQueued = true;
  requestAnimationFrame(() => {
    logFlushQueued = false;
    flushLog();
  });
}

function flushLog() {
  if (!logBuffer.length) return;
  const area = $("#log-area");
  const frag = document.createDocumentFragment();
  for (const line of logBuffer) {
    const div = document.createElement("div");
    div.className = "line";
    if (/error|failed/i.test(line)) div.className = "line err";
    else if (/warning|warn/i.test(line)) div.className = "line warn";
    div.textContent = line;
    frag.appendChild(div);
  }
  logBuffer.length = 0;
  area.appendChild(frag);
  while (area.children.length > 500) area.removeChild(area.firstChild);
  area.scrollTop = area.scrollHeight;
}

// ---------- env + settings ----------
async function refreshEnv(force = false) {
  setStatus(t("status.checkingEnv"));
  try {
    state.env = await invoke("check_environment", { force });
    renderEnv();
    setModelStatus(`${t("env.cudaLabel")} ${state.env.cuda ? t("env.cudaOn") : t("env.cudaOff")} · ${t("env.libsLabel")} ${state.env.pythonLibraries ? t("env.libsOk") : t("env.libsMissing")}`);
    setStatus(t("status.envComplete"));
  } catch (e) {
    setStatus(`${t("status.envFailed")} ${e}`);
  }
}

function renderEnv() {
  const e = state.env;
  if (!e) { $("#env-status").innerHTML = `<div class="empty">${t("env.runCheck")}</div>`; return; }
  const row = (label, ok, val) =>
    `<div class="env-row"><span>${label}</span><span class="${ok ? "env-ok" : "env-bad"}">${esc(val ?? (ok ? t("env.ok") : t("env.missing")))}</span></div>`;
  const libs = e.libraries || {};
  const libNames = ["torch", "torchvision", "transformers", "numpy", "soundfile", "funasr"];
  const missing = libNames.filter((k) => String(libs[k] ?? "").indexOf("missing") !== -1);
  const libVal = missing.length
    ? `${t("env.missingLibs")} ${missing.join(", ")}`
    : libs.torch + (libs.funasr ? " · funasr" : "");
  const dl = e.downloadLibraries || {};
  const dlNames = Object.keys(dl);
  const dlMissing = dlNames.filter((k) => String(dl[k] ?? "").indexOf("missing") !== -1);
  const dlVal = !dlNames.length
    ? t("env.missing")
    : dlMissing.length
      ? `${t("env.downloadLibsMissing")}: ${dlMissing.join(", ")}`
      : dlNames.map((k) => `${k} ${dl[k]}`).join(" · ");
  $("#env-status").innerHTML = [
    row(t("env.python"), e.python.ok, `${e.python.command} ${e.python.version}`),
    row(t("env.ffmpeg"), e.ffmpeg),
    row(t("env.cuda"), e.cuda),
    row(t("env.pythonLibs"), e.pythonLibraries, libVal),
    row(t("env.downloadLibs"), dlNames.length > 0 && !dlMissing.length, dlVal),
  ].join("");
}

async function loadSettings() {
  state.config = await invoke("get_config");
  $("#cfg-videoneko").value = state.config.videonekoModelDir || "";
  $("#cfg-asr-type").value = state.config.asrType || "sensevoice-small";
  $("#cfg-asr-source").value = state.config.asrSource || "huggingface";
  $("#cfg-asr-id").value = state.config.asrModelId || defaultModelId(state.config.asrType, state.config.asrSource);
  $("#cfg-asr-dir").value = state.config.asrModelDir || "";
  $("#cfg-asr-language").value = state.config.asrLanguage || "";
  $("#cfg-vad-source").value = state.config.vadSource || "huggingface";
  $("#cfg-vad-id").value = state.config.vadModelId || defaultModelId("fsmn-vad", state.config.vadSource);
  $("#cfg-vad-dir").value = state.config.vadModelDir || "";
  $("#cfg-spk-model-enabled").checked = !!(state.config.spkEnabled && (state.config.spkModelDir || "").trim());
  $("#cfg-spk-source").value = state.config.spkSource || "huggingface";
  $("#cfg-spk-id").value = state.config.spkModelId || defaultModelId("cam++", state.config.spkSource);
  $("#cfg-spk-dir").value = state.config.spkModelDir || "";
  $("#cfg-qwen3-base").value = state.config.qwen3BaseUrl || "";
  $("#cfg-qwen3-key").value = state.config.qwen3ApiKey || "";
  $("#cfg-qwen3-model").value = state.config.qwen3Model || "";
  $("#cfg-qwen3-language").value = state.config.qwen3Language || "";
  $("#cfg-api-base").value = state.config.apiBaseUrl || "";
  $("#cfg-api-key").value = state.config.apiKey || "";
  $("#cfg-api-model").value = state.config.apiModel || "";
  $("#cfg-api-max-tokens").value = state.config.apiMaxTokens || 8192;
  $("#cfg-api-temperature").value = state.config.apiTemperature ?? 0.4;
  $("#cfg-api-top-p").value = state.config.apiTopP ?? 1.0;
  $("#cfg-api-thinking").checked = !!state.config.apiThinking;
  $("#cfg-ollama-base").value = state.config.ollamaBaseUrl || "";
  $("#cfg-ollama-model").value = state.config.ollamaModel || "";
  $("#cfg-ollama-thinking").checked = !!state.config.ollamaThinking;
  $("#cfg-llamacpp-base").value = state.config.llamacppBaseUrl || "";
  $("#cfg-llamacpp-model").value = state.config.llamacppModel || "";
  $("#cfg-llamacpp-thinking").checked = !!state.config.llamacppThinking;
  $("#cfg-language").value = state.config.language || "";
  setSpeakerSettings();
  setLanguage(state.config.language || "");
  setEngine(state.config.engine || "api");
  setQuality(state.config.downloadQuality || 720);
  setAsrType(state.config.asrType || "sensevoice-small");
  setHubRows();
  setSpkModelSettings();
  renderModelReport(state.modelReport);
  $("#cfg-cookie-browser").value = ["", "firefox", "chrome", "edge"].includes(state.config.cookieBrowser)
    ? state.config.cookieBrowser
    : "";
  try {
    $("#cfg-prompt").value = await invoke("get_prompt");
  } catch {
    $("#cfg-prompt").value = "";
  }
}

function setSpeakerSettings() {
  const enabled = !!(state.config.speakerName || "").trim();
  $("#cfg-spk-enabled").checked = enabled;
  $("#cfg-spk-name").value = enabled ? state.config.speakerName : "";
  $("#spk-settings").classList.toggle("hidden", !enabled);
  renderSpkFileStatus();
}

function renderSpkFileStatus() {
  const el = $("#spk-file");
  if (state.speakerWav) {
    el.textContent = state.speakerWav.split(/[\\/]/).pop() + " " + t("settings.speakerFileNew");
  } else if ((state.config?.speakerRef || "").trim()) {
    el.textContent = t("settings.speakerFileCurrent") + " " + state.config.speakerRef;
  } else {
    el.textContent = t("settings.speakerFileNone");
  }
}

function setEngine(engine) {
  document.querySelectorAll('input[name="engine"]').forEach((r) => {
    r.checked = r.value === engine;
  });
  $("#api-settings").classList.toggle("hidden", engine !== "api");
  $("#ollama-settings").classList.toggle("hidden", engine !== "ollama");
  $("#llamacpp-settings").classList.toggle("hidden", engine !== "llamacpp");
}

function setQuality(quality) {
  const q = [360, 480, 720, 1080].includes(Number(quality)) ? String(quality) : "720";
  document.querySelectorAll('input[name="quality"]').forEach((r) => {
    r.checked = r.value === q;
  });
}

// ---------- FunASR model configuration ----------
const DEFAULT_MODEL_IDS = {
  "sensevoice-small": { huggingface: "FunAudioLLM/SenseVoiceSmall", modelscope: "iic/SenseVoiceSmall" },
  "fun-asr-nano": { huggingface: "FunAudioLLM/Fun-ASR-Nano-2512", modelscope: "FunAudioLLM/Fun-ASR-Nano-2512" },
  "paraformer-zh-streaming": {
    huggingface: "funasr/paraformer-zh-streaming",
    modelscope: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
  },
  "fsmn-vad": { huggingface: "funasr/fsmn-vad", modelscope: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" },
  "cam++": { huggingface: "funasr/campplus", modelscope: "iic/speech_campplus_sv_zh-cn_16k-common" },
};

function defaultModelId(kind, source) {
  const entry = DEFAULT_MODEL_IDS[kind] || {};
  return entry[source === "modelscope" ? "modelscope" : "huggingface"] || "";
}

function setAsrType(type) {
  const isApi = type === "qwen3-api";
  $("#asr-local-settings").classList.toggle("hidden", isApi);
  $("#asr-api-settings").classList.toggle("hidden", !isApi);
}

function setHubRows() {
  const pairs = [
    ["#cfg-asr-source", "#asr-download-row"],
    ["#cfg-vad-source", "#vad-download-row"],
    ["#cfg-spk-source", "#spk-download-row"],
  ];
  for (const [sel, row] of pairs) {
    $(row).classList.toggle("hidden", $(sel).value === "local");
  }
}

function setSpkModelSettings() {
  const enabled = $("#cfg-spk-model-enabled").checked;
  $("#spk-model-settings").classList.toggle("hidden", !enabled);
}

function slotStatusText(item) {
  if (!item) return "";
  if (item.skipped) return t("settings.skipped");
  if (item.ok) return t("settings.valid");
  return `${t("settings.invalid")}: ${(item.errors || []).join("; ")}`;
}

function renderModelReport(report) {
  if (!report) {
    ["#asr-status", "#vad-status", "#spk-status"].forEach((sel) => { $(sel).textContent = ""; });
    return;
  }
  const items = report.items || {};
  $("#asr-status").textContent = slotStatusText(items.asr);
  $("#vad-status").textContent = slotStatusText(items.vad);
  $("#spk-status").textContent = slotStatusText(items.spk);
}

function browseDir(inputSel) {
  open({ directory: true }).then((p) => { if (p) $(inputSel).value = p; });
}

async function downloadModel(kind, sourceSel, idInput, dirInput, statusEl) {
  const source = $(sourceSel).value;
  const modelId = $(idInput).value.trim();
  if (source === "local") {
    $(statusEl).textContent = t("settings.downloadNeedsHub");
    return;
  }
  if (!modelId) {
    $(statusEl).textContent = t("settings.downloadFailed");
    return;
  }
  $(statusEl).textContent = t("settings.downloading");
  try {
    const r = await invoke("download_model", { kind, source, modelId });
    $(dirInput).value = r.path || "";
    $(statusEl).textContent = t("settings.downloadDone");
  } catch (e) {
    $(statusEl).textContent = `${t("settings.downloadFailed")} ${e}`;
  }
}

async function validateModels() {
  const config = collectConfig();
  $("#save-status").textContent = t("settings.validating");
  try {
    const report = await invoke("validate_model_config", { config });
    state.modelReport = report;
    renderModelReport(report);
    $("#save-status").textContent = report.ok ? t("settings.modelsOk") : t("settings.invalid");
    setTimeout(() => ($("#save-status").textContent = ""), 2500);
  } catch (e) {
    $("#save-status").textContent = `${t("settings.saveError")} ${e}`;
  }
}

async function testAsrApi() {
  const el = $("#asr-api-status");
  el.textContent = t("settings.testing");
  try {
    const r = await invoke("test_asr_connection", {
      baseUrl: $("#cfg-qwen3-base").value.trim(),
      apiKey: $("#cfg-qwen3-key").value.trim(),
      model: $("#cfg-qwen3-model").value.trim(),
      language: $("#cfg-qwen3-language").value.trim(),
    });
    el.textContent = r && r.warning === "probe_audio_rejected"
      ? t("settings.connectionOkProbe")
      : t("settings.connectionOk");
  } catch (e) {
    el.textContent = `${t("settings.connectionFailed")} ${e}`;
  }
}

function collectConfig() {
  const engine = document.querySelector('input[name="engine"]:checked')?.value || "api";
  const qualityInput = document.querySelector('input[name="quality"]:checked');
  const spkEnabled = $("#cfg-spk-enabled").checked;
  const spkName = spkEnabled ? $("#cfg-spk-name").value.trim() : "";
  return {
    videonekoModelDir: $("#cfg-videoneko").value.trim(),
    engine,
    downloadQuality: qualityInput ? parseInt(qualityInput.value) || 720 : 720,
    cookieBrowser: $("#cfg-cookie-browser").value,
    language: $("#cfg-language").value,
    customPrompt: $("#cfg-prompt").value,
    speakerName: spkName,
    // keep the stored reference unless a new WAV was picked
    speakerRef: spkEnabled && !state.speakerWav ? state.config?.speakerRef || "" : "",
    funasrModelsDir: state.config?.funasrModelsDir || "",
    asrType: $("#cfg-asr-type").value,
    asrSource: $("#cfg-asr-source").value,
    asrModelId: $("#cfg-asr-id").value.trim(),
    asrModelDir: $("#cfg-asr-dir").value.trim(),
    asrLanguage: $("#cfg-asr-language").value.trim(),
    vadSource: $("#cfg-vad-source").value,
    vadModelId: $("#cfg-vad-id").value.trim(),
    vadModelDir: $("#cfg-vad-dir").value.trim(),
    spkEnabled: $("#cfg-spk-model-enabled").checked,
    spkSource: $("#cfg-spk-source").value,
    spkModelId: $("#cfg-spk-id").value.trim(),
    spkModelDir: $("#cfg-spk-dir").value.trim(),
    qwen3BaseUrl: $("#cfg-qwen3-base").value.trim(),
    qwen3ApiKey: $("#cfg-qwen3-key").value.trim(),
    qwen3Model: $("#cfg-qwen3-model").value.trim(),
    qwen3Language: $("#cfg-qwen3-language").value.trim(),
    apiBaseUrl: $("#cfg-api-base").value.trim(),
    apiKey: $("#cfg-api-key").value.trim(),
    apiModel: $("#cfg-api-model").value.trim(),
    apiMaxTokens: parseInt($("#cfg-api-max-tokens").value) || 8192,
    apiTemperature: parseFloat($("#cfg-api-temperature").value) || 0.4,
    apiTopP: parseFloat($("#cfg-api-top-p").value) ?? 1.0,
    apiThinking: $("#cfg-api-thinking").checked,
    ollamaBaseUrl: $("#cfg-ollama-base").value.trim(),
    ollamaModel: $("#cfg-ollama-model").value.trim(),
    ollamaThinking: $("#cfg-ollama-thinking").checked,
    llamacppBaseUrl: $("#cfg-llamacpp-base").value.trim(),
    llamacppModel: $("#cfg-llamacpp-model").value.trim(),
    llamacppThinking: $("#cfg-llamacpp-thinking").checked,
  };
}

async function saveSettings() {
  const spkEnabled = $("#cfg-spk-enabled").checked;
  const spkName = spkEnabled ? $("#cfg-spk-name").value.trim() : "";
  if (spkEnabled && !spkName) {
    $("#save-status").textContent = t("settings.speakerNameRequired");
    return;
  }
  // A speaker needs a reference WAV: either one picked in this session or the
  // previously imported file (kept on the backend until the speaker changes).
  if (spkEnabled && !state.speakerWav && !(state.config?.speakerRef || "").trim()) {
    $("#save-status").textContent = t("settings.speakerWavRequired");
    return;
  }
  const config = collectConfig();
  try {
    await invoke("save_config", { config, speakerWav: state.speakerWav || null });
    state.speakerWav = "";
    // re-fetch so backend-normalized fields (e.g. the stored reference name) show up
    state.config = await invoke("get_config");
    renderSpkFileStatus();
    $("#save-status").textContent = t("settings.saved");
    setTimeout(() => ($("#save-status").textContent = ""), 2000);
  } catch (e) {
    $("#save-status").textContent = `${t("settings.saveError")} ${e}`;
  }
}

// ---------- results ----------
async function refreshResults() {
  try {
    state.results = await invoke("list_results");
  } catch {
    state.results = [];
  }
  renderResultsList();
}

function renderResultsList() {
  const list = $("#results-list");
  if (!state.results.length) {
    list.innerHTML = `<div class="empty">${t("results.noSummaries")}</div>`;
    $("#results-detail").innerHTML = `<div class="empty">${t("results.selectHint")}</div>`;
    return;
  }
  renderResultList(state.results);
}

function renderResultList(results) {
  const list = $("#results-list");
  if (!results.length) {
    list.innerHTML = `<div class="empty">${t("results.noMatch")}</div>`;
    return;
  }
  list.innerHTML = results
    .map((r) => `<div class="res-item" data-stem="${esc(r.stem)}">
        <div class="res-title">${esc(r.stem)}</div>
        <div class="res-date">${fmtDate(r.modified)} · ${fmtSize(r.size)}</div>
        ${r.snippet !== undefined ? `<div class="res-snippet">${esc(r.snippet)}</div>` : ""}
      </div>`)
    .join("");
  list.querySelectorAll(".res-item").forEach((el) => {
    el.addEventListener("click", () => {
      document.querySelectorAll(".res-item").forEach((x) => x.classList.remove("active"));
      el.classList.add("active");
      openResult(el.dataset.stem);
    });
  });
}

async function doSearch() {
  const query = $("#results-search").value.trim();
  const regex = $("#results-regex").checked;
  if (!query) {
    refreshResults();
    $("#results-search-status").textContent = "";
    return;
  }
  try {
    const matches = await invoke("search_results", { query, useRegex: regex });
    renderResultList(matches);
    $("#results-search-status").textContent = t("results.matchCount", { n: matches.length });
  } catch (e) {
    $("#results-search-status").textContent = `${t("results.searchError")} ${e}`;
  }
}

async function clearSearch() {
  $("#results-search").value = "";
  $("#results-search-status").textContent = "";
  refreshResults();
}

async function openResult(stem) {
  try {
    const r = await invoke("read_result", { stem });
    state.activeResult = r;
    const summaryHtml = renderSummaryHtml(r.summary, r.thinking);
    $("#results-detail").innerHTML = `
      <div class="result-head">
        <h2>${esc(stem)}</h2>
        <div class="result-tools">
          <button id="btn-resummarize" class="btn small" title="${t("results.resummarize")}">${t("results.resummarize")}</button>
          <button id="btn-view-asr" class="btn ghost small">${t("results.showAsr")}</button>
          <button id="btn-view-visual" class="btn ghost small">${t("results.showVisual")}</button>
          <button id="btn-copy" class="btn ghost small">${t("results.copy")}</button>
          <button id="btn-export" class="btn ghost small">${t("results.export")}</button>
          <button id="btn-delete" class="btn danger small">${t("results.delete")}</button>
        </div>
      </div>
      <span id="resummarize-status" class="hint"></span>
      <div class="markdown-body">${summaryHtml}</div>
      <pre id="detail-raw" class="hidden" style="margin-top:10px;background:var(--code-bg);padding:10px;border-radius:6px;overflow:auto;max-height:400px;"></pre>`;
    $("#btn-copy").addEventListener("click", async () => {
      await navigator.clipboard.writeText(r.summary);
      $("#btn-copy").textContent = t("results.copied");
      setTimeout(() => ($("#btn-copy").textContent = t("results.copy")), 1500);
    });
    $("#btn-export").addEventListener("click", async () => {
      const dest = await save({
        defaultPath: `${stem}.summary.md`,
        filters: [{ name: "Markdown", extensions: ["md"] }],
      });
      if (dest) {
        try { await invoke("export_result", { stem, dest }); alert(`${t("results.exported")} ${dest}`); }
        catch (e) { alert(`${t("results.exportFailed")} ${e}`); }
      }
    });
    $("#btn-resummarize").addEventListener("click", async () => {
      $("#resummarize-status").textContent = t("results.resummarizing");
      $("#btn-resummarize").disabled = true;
      try {
        await invoke("re_summarize", { stem });
      } catch (e) {
        $("#resummarize-status").textContent = `${t("results.failed")} ${e}`;
        $("#btn-resummarize").disabled = false;
      }
    });
    $("#btn-delete").addEventListener("click", async () => {
      if (!confirm(t("results.deleteConfirm", { stem }))) return;
      try {
        await invoke("delete_result", { stem });
        if (state.activeResult && state.activeResult.stem === stem) state.activeResult = null;
        refreshResults();
        $("#results-detail").innerHTML = `<div class="empty">${t("results.deleted")}</div>`;
      } catch (e) {
        alert(`${t("results.deleteFailed")} ${e}`);
      }
    });
    $("#btn-view-asr").addEventListener("click", () => {
      const raw = $("#detail-raw");
      raw.classList.toggle("hidden");
      raw.textContent = r.asr || t("results.noAsr");
    });
    $("#btn-view-visual").addEventListener("click", () => {
      const raw = $("#detail-raw");
      raw.classList.toggle("hidden");
      raw.textContent = r.visual || t("results.noVisual");
    });
  } catch (e) {
    $("#results-detail").innerHTML = `<div class="empty">${t("results.loadFailed")} ${esc(e)}</div>`;
  }
}

function renderSummaryHtml(summary, thinking) {
  const parts = [];
  let body = summary || "";
  if (!thinking || !thinking.trim()) {
    const lines = body.split(/\r?\n/);
    const idx = lines.findIndex((l) => /^\s*\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\]/.test(l));
    if (idx > 0) {
      thinking = lines.slice(0, idx).join("\n").trim();
      body = lines.slice(idx).join("\n");
    }
  }
  if (thinking && thinking.trim()) {
    parts.push(`<details class="think-block">
      <summary>${t("results.thinking")}</summary>
      <div class="think-body">${renderMarkdown(thinking)}</div>
    </details>`);
  }
  if (!body || !body.trim()) {
    parts.push(`<div class="empty">${t("results.noSummary")}</div>`);
    return parts.join("");
  }
  const lines = body.split(/\r?\n/);
  let html = "";
  for (const line of lines) {
    const m = line.match(/^\s*(\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\])\s*(.*)$/);
    if (m) {
      html += `<div class="summary-entry"><span class="entry-time">${esc(m[1])}</span><span class="entry-text">${renderMarkdown(m[2])}</span></div>`;
    } else if (line.trim()) {
      html += renderMarkdown(line);
    }
  }
  parts.push(html);
  return parts.join("");
}

function renderMarkdown(text) {
  try {
    return marked.parse(text);
  } catch {
    return esc(text);
  }
}

// ---------- events from backend ----------
async function wireEvents() {
  await listen("pipeline://log", (e) => {
    appendLog(e.payload.line);
  });
  await listen("model://progress", (e) => {
    const { kind, progress } = e.payload || {};
    const el = { asr: "#asr-status", vad: "#vad-status", spk: "#spk-status" }[kind];
    if (el) $(el).textContent = `${t("settings.downloading")} ${progress}%`;
    if (!$("#setup-modal").classList.contains("hidden")) {
      $("#setup-funasr-status").textContent = `${t("settings.downloading")} ${progress}%`;
    }
  });
  await listen("pipeline://stage", (e) => {
    const { itemId, stage, progress, part, totalParts } = e.payload;
    updateItemStage(itemId, stage, progress, part, totalParts);
  });
  await listen("pipeline://start", () => {
    state.running = true;
    renderRunControls();
    setStatus(t("status.running"));
  });
  await listen("pipeline://done", (e) => {
    const { itemId, ok, error, cancelled } = e.payload;
    if (ok) {
      appendLog(`[${itemId}] ${t("log.finished")}`);
    } else {
      appendLog(`[${itemId}] ${cancelled ? t("log.cancelled") : `${t("log.failed")} ${error ?? ""}`}`);
    }
    refreshQueue();
  });
  await listen("pipeline://finished", async () => {
    state.running = false;
    renderRunControls();
    setStatus(t("status.finished"));
    refreshQueue();
    refreshResults();
    if (state.activeResult) {
      const st = state.activeResult.stem;
      $("#resummarize-status").textContent = t("results.done");
      $("#btn-resummarize").disabled = false;
      const r = await invoke("read_result", { stem: st }).catch(() => null);
      if (r) {
        state.activeResult = r;
        openResult(st);
      }
    }
  });
}

function updateItemStage(itemId, stage, progress, part, totalParts) {
  const item = state.queue.find((i) => i.id === itemId);
  if (item) {
    if (!item.stageProgress || item.stageProgress.length !== 4) {
      item.stageProgress = [0, 0, 0, 0];
    }
    let pct = progress;
    const tp = totalParts || item.totalParts || 1;
    if (tp > 1 && (stage === 2 || stage === 3)) {
      const p = part || item.currentPart || 1;
      pct = Math.round(((p - 1) + progress / 100) / tp * 100);
    }
    item.stageProgress[stage - 1] = pct;
    if (part) item.currentPart = part;
    if (totalParts) item.totalParts = totalParts;
    item.status = "Running";
    scheduleQueueRender();
  }
}

// ---------- actions ----------
async function addUrl() {
  const url = $("#url-input").value.trim();
  if (!url) return;
  try {
    await invoke("add_url", { url });
    $("#url-input").value = "";
    refreshQueue();
  } catch (e) {
    setStatus(`${t("status.addUrlFailed")} ${e}`);
  }
}

async function addFile() {
  const files = await open({
    multiple: true,
    filters: [{ name: "Video", extensions: ["mp4", "mkv", "mov", "webm", "avi", "flv"] }],
  });
  if (!files) return;
  for (const f of Array.isArray(files) ? files : [files]) {
    try { await invoke("add_local_file", { path: f }); } catch (e) { setStatus(`${t("status.addFileFailed")} ${e}`); }
  }
  refreshQueue();
}

async function startPipeline() {
  try {
    await invoke("start_pipeline");
  } catch (e) {
    setStatus(`${t("status.startFailed")} ${e}`);
  }
}

async function stopPipeline() {
  await invoke("stop_pipeline");
  setStatus(t("status.stopRequested"));
}

function browseVideoneko() {
  open({ directory: true }).then((p) => { if (p) $("#cfg-videoneko").value = p; });
}

function browseSpeakerWav() {
  open({
    multiple: false,
    filters: [{ name: "WAV audio", extensions: ["wav"] }],
  }).then((p) => {
    if (p) {
      state.speakerWav = p;
      renderSpkFileStatus();
    }
  });
}

async function testEngine(engine, statusEl) {
  let base, key, model;
  if (engine === "api") {
    base = $("#cfg-api-base").value.trim();
    key = $("#cfg-api-key").value.trim();
    model = $("#cfg-api-model").value.trim();
  } else if (engine === "ollama") {
    base = $("#cfg-ollama-base").value.trim();
    key = "ollama";
    model = $("#cfg-ollama-model").value.trim();
  } else {
    base = $("#cfg-llamacpp-base").value.trim();
    key = "llamacpp";
    model = $("#cfg-llamacpp-model").value.trim();
  }
  statusEl.textContent = t("settings.testing");
  try {
    const r = await invoke("test_api_connection", { baseUrl: base, apiKey: key, model });
    statusEl.textContent = r.ok ? t("settings.connectionOk") : t("settings.connectionFailed");
  } catch (e) {
    statusEl.textContent = `${t("settings.connectionFailed")} ${e}`;
  }
}

async function testApi() {
  await testEngine("api", $("#api-test-status"));
}

// ---------- setup wizard ----------
async function checkSetup() {
  try {
    const s = await invoke("get_setup_status");
    if (s.firstLaunch || s.needsVideoneko || s.needsFunasr) {
      $("#setup-modal").classList.remove("hidden");
      const cfg = state.config || {};
      $("#setup-videoneko").value = s.videonekoModelDir || "";
      $("#setup-asr-type").value = cfg.asrType || "sensevoice-small";
      $("#setup-asr-source").value = cfg.asrSource || "huggingface";
      $("#setup-asr-id").value = cfg.asrModelId || defaultModelId(cfg.asrType || "sensevoice-small", $("#setup-asr-source").value);
      $("#setup-asr-dir").value = cfg.asrModelDir || "";
      $("#setup-vad-dir").value = cfg.vadModelDir || "";
      $("#setup-qwen3-base").value = cfg.qwen3BaseUrl || "";
      $("#setup-qwen3-key").value = cfg.qwen3ApiKey || "";
      $("#setup-qwen3-model").value = cfg.qwen3Model || "";
      setSetupAsrType($("#setup-asr-type").value);
      const cur = (state.config && state.config.engine) || "api";
      setSetupEngine(["api", "ollama", "llamacpp"].includes(cur) ? cur : "api");
    }
  } catch {
    // ignore
  }
}

function setSetupAsrType(type) {
  const isApi = type === "qwen3-api";
  $("#setup-asr-local").classList.toggle("hidden", isApi);
  $("#setup-asr-api").classList.toggle("hidden", !isApi);
  $("#setup-asr-dir-row").classList.toggle("hidden", isApi);
  $("#setup-vad-row").classList.toggle("hidden", isApi);
}

function setSetupEngine(engine) {
  document.querySelectorAll('input[name="setup-engine"]').forEach((r) => {
    r.checked = r.value === engine;
  });
  $("#setup-api-row").style.display = engine === "api" ? "flex" : "none";
  $("#setup-ollama-row").style.display = engine === "ollama" ? "flex" : "none";
  $("#setup-llamacpp-row").style.display = engine === "llamacpp" ? "flex" : "none";
}

function setupValues() {
  const engine = document.querySelector('input[name="setup-engine"]:checked')?.value || "api";
  return {
    videonekoModelDir: $("#setup-videoneko").value.trim(),
    engine,
    apiBaseUrl: $("#setup-api-base").value.trim(),
    apiKey: $("#setup-api-key").value.trim(),
    apiModel: $("#setup-api-model").value.trim(),
    ollamaBaseUrl: $("#setup-ollama-base").value.trim(),
    ollamaModel: $("#setup-ollama-model").value.trim(),
    llamacppBaseUrl: $("#setup-llamacpp-base").value.trim(),
    llamacppModel: $("#setup-llamacpp-model").value.trim(),
    asrType: $("#setup-asr-type").value,
    asrSource: $("#setup-asr-source").value,
    asrModelId: $("#setup-asr-id").value.trim(),
    asrModelDir: $("#setup-asr-dir").value.trim(),
    vadModelDir: $("#setup-vad-dir").value.trim(),
    qwen3BaseUrl: $("#setup-qwen3-base").value.trim(),
    qwen3ApiKey: $("#setup-qwen3-key").value.trim(),
    qwen3Model: $("#setup-qwen3-model").value.trim(),
  };
}

async function finishSetup() {
  const v = setupValues();
  const cfg = { ...(state.config || {}), ...v };
  const statusEl = $("#setup-funasr-status");
  $("#setup-finish").disabled = true;
  statusEl.textContent = t("settings.validating");
  try {
    const report = await invoke("validate_model_config", { config: cfg });
    if (!report.ok) {
      statusEl.textContent = `${t("setup.funasrIncomplete")} ${(report.errors || []).join("; ")}`;
      return;
    }
    await invoke("save_config", { config: cfg });
    state.config = await invoke("get_config");
    $("#setup-modal").classList.add("hidden");
    loadSettings();
    setStatus(t("status.setupComplete"));
  } catch (e) {
    statusEl.textContent = `${t("status.setupFailed")} ${e}`;
  } finally {
    $("#setup-finish").disabled = false;
  }
}

// ---------- apply language to the DOM ----------
function applyLanguage() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  renderRunControls();
  renderQueue();
  renderEnv();
  renderSpkFileStatus();
  renderModelReport(state.modelReport);
  renderResultsList();
  if (state.activeResult) openResult(state.activeResult.stem);
}

// ---------- wire up DOM ----------
function init() {
  $("#btn-add-url").addEventListener("click", addUrl);
  $("#url-input").addEventListener("keydown", (e) => { if (e.key === "Enter") addUrl(); });
  $("#btn-add-file").addEventListener("click", addFile);
  $("#btn-run").addEventListener("click", () => {
    if (state.running) stopPipeline();
    else startPipeline();
  });
  $("#btn-clear").addEventListener("click", async () => { await invoke("clear_queue"); refreshQueue(); });
  $("#btn-recheck").addEventListener("click", () => refreshEnv(true));
  $("#btn-browse-videoneko").addEventListener("click", browseVideoneko);
  // FunASR model configuration
  $("#cfg-asr-type").addEventListener("change", () => {
    const type = $("#cfg-asr-type").value;
    setAsrType(type);
    $("#cfg-asr-id").value = defaultModelId(type, $("#cfg-asr-source").value);
  });
  $("#cfg-asr-source").addEventListener("change", () => {
    $("#cfg-asr-id").value = defaultModelId($("#cfg-asr-type").value, $("#cfg-asr-source").value);
    setHubRows();
  });
  $("#cfg-vad-source").addEventListener("change", () => {
    $("#cfg-vad-id").value = defaultModelId("fsmn-vad", $("#cfg-vad-source").value);
    setHubRows();
  });
  $("#cfg-spk-source").addEventListener("change", () => {
    $("#cfg-spk-id").value = defaultModelId("cam++", $("#cfg-spk-source").value);
    setHubRows();
  });
  $("#cfg-spk-model-enabled").addEventListener("change", setSpkModelSettings);
  $("#btn-download-asr").addEventListener("click", () => downloadModel("asr", "#cfg-asr-source", "#cfg-asr-id", "#cfg-asr-dir", "#asr-status"));
  $("#btn-download-vad").addEventListener("click", () => downloadModel("vad", "#cfg-vad-source", "#cfg-vad-id", "#cfg-vad-dir", "#vad-status"));
  $("#btn-download-spk").addEventListener("click", () => downloadModel("spk", "#cfg-spk-source", "#cfg-spk-id", "#cfg-spk-dir", "#spk-status"));
  $("#btn-browse-asr").addEventListener("click", () => browseDir("#cfg-asr-dir"));
  $("#btn-browse-vad").addEventListener("click", () => browseDir("#cfg-vad-dir"));
  $("#btn-browse-spk").addEventListener("click", () => browseDir("#cfg-spk-dir"));
  $("#btn-validate-asr").addEventListener("click", validateModels);
  $("#btn-validate-vad").addEventListener("click", validateModels);
  $("#btn-validate-spk").addEventListener("click", validateModels);
  $("#btn-test-asr-api").addEventListener("click", testAsrApi);
  $("#setup-asr-type").addEventListener("change", () => {
    setSetupAsrType($("#setup-asr-type").value);
    $("#setup-asr-id").value = defaultModelId($("#setup-asr-type").value, $("#setup-asr-source").value);
  });
  $("#setup-asr-source").addEventListener("change", () => {
    $("#setup-asr-id").value = defaultModelId($("#setup-asr-type").value, $("#setup-asr-source").value);
  });
  $("#setup-download-asr").addEventListener("click", () =>
    downloadModel("asr", "#setup-asr-source", "#setup-asr-id", "#setup-asr-dir", "#setup-funasr-status"));
  $("#setup-download-vad").addEventListener("click", async () => {
    const source = $("#setup-asr-source").value;
    const statusEl = $("#setup-funasr-status");
    if (source === "local") {
      statusEl.textContent = t("settings.downloadNeedsHub");
      return;
    }
    statusEl.textContent = t("settings.downloading");
    try {
      const r = await invoke("download_model", { kind: "vad", source, modelId: defaultModelId("fsmn-vad", source) });
      $("#setup-vad-dir").value = r.path || "";
      statusEl.textContent = t("settings.downloadDone");
    } catch (e) {
      statusEl.textContent = `${t("settings.downloadFailed")} ${e}`;
    }
  });
  $("#setup-browse-asr").addEventListener("click", () => browseDir("#setup-asr-dir"));
  $("#setup-browse-vad").addEventListener("click", () => browseDir("#setup-vad-dir"));
  $("#cfg-spk-enabled").addEventListener("change", () => {
    $("#spk-settings").classList.toggle("hidden", !$("#cfg-spk-enabled").checked);
  });
  $("#btn-browse-spk").addEventListener("click", browseSpeakerWav);
  $("#btn-test-api").addEventListener("click", testApi);
  $("#btn-test-ollama").addEventListener("click", () => testEngine("ollama", $("#ollama-test-status")));
  $("#btn-test-llamacpp").addEventListener("click", () => testEngine("llamacpp", $("#llamacpp-test-status")));
  $("#btn-save-settings").addEventListener("click", saveSettings);
  $("#cfg-language").addEventListener("change", () => setLanguage($("#cfg-language").value));
  $("#btn-reset-prompt").addEventListener("click", async () => {
    try {
      await invoke("reset_prompt");
      const defaultPrompt = await invoke("get_prompt");
      $("#cfg-prompt").value = defaultPrompt;
      $("#save-status").textContent = t("settings.promptReset");
      setTimeout(() => ($("#save-status").textContent = ""), 2000);
    } catch (e) {
      $("#save-status").textContent = `${t("settings.resetFailed")} ${e}`;
    }
  });
  document.querySelectorAll('input[name="engine"]').forEach((r) =>
    r.addEventListener("change", () => setEngine(r.value))
  );
  document.querySelectorAll('input[name="setup-engine"]').forEach((r) =>
    r.addEventListener("change", () => setSetupEngine(r.value))
  );
  $("#setup-browse-videoneko").addEventListener("click", () => {
    open({ directory: true }).then((p) => { if (p) $("#setup-videoneko").value = p; });
  });
  $("#setup-finish").addEventListener("click", finishSetup);
  $("#setup-skip").addEventListener("click", () => {
    $("#setup-modal").classList.add("hidden");
  });
  $("#btn-results-search").addEventListener("click", doSearch);
  $("#results-search").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  $("#btn-results-clear").addEventListener("click", clearSearch);
}

(async function main() {
  await applyOsTheme();
  init();
  applyLanguage();
  await wireEvents();
  await loadSettings();
  await refreshEnv();
  await refreshQueue();
  refreshResults();
  await checkSetup();
})();
