# HELEN AI 6.0 — Qwen2.5 1.5B Bundled Offline Assistant

A Hebrew-first Android personal assistant powered by **Qwen2.5-1.5B-Instruct INT8** through Google's LiteRT-LM runtime.

## Built-in AI model
- The approximately **1.6 GB** Qwen2.5-1.5B INT8 LiteRT-LM model is embedded directly in the APK.
- No separate model download is required after installing the app. During first launch, HELEN copies the packaged model from APK assets to private app storage because LiteRT-LM loads the model from a filesystem path.
- Chat and sentence suggestions then run locally on the phone without an internet connection.
- The app tries the GPU backend first, then falls back to CPU if GPU initialization is unavailable.
- Published Android measurements on a Samsung S25 Ultra for the matching INT8 model reported approximately 30.9 tokens/second on GPU and 34.3 tokens/second on CPU for the 1280-token benchmark; the 4096-token model artifact used by this app measured approximately 27.3 on GPU and 26.1 on CPU. Results vary by phone.
- Published peak memory use for the 4096-token benchmark was around 1.9 GB on GPU and 2.2 GB on CPU. A device with limited free RAM may close the app or require a different backend.

## Features
- Hebrew-first AI chat and AI-generated sentence completion
- Multi-command interpretation for phone settings and app launch
- Accessibility-assisted tap, type, scroll, read screen and screenshot (after explicit Android opt-in)
- Reminders, alarms, timers, notes, tasks, shopping list, expenses and water tracking
- Personal routines and local text knowledge import
- Dark/light theme, text scaling, speech rate, chat history and model storage management

## Model provenance and license
- Model repository: `litert-community/Qwen2.5-1.5B-Instruct`
- Bundled file: `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- The model repository declares Apache-2.0 licensing.

## Android limits
Wi-Fi/Bluetooth radios are handled through Android settings; phone calls and WhatsApp messages require user confirmation. Speech recognition availability depends on the device's recognition service. Some screen-control actions require explicit accessibility opt-in.

## Build
GitHub Actions downloads the public model at build time, packages it into `app/src/main/assets`, builds the APK, verifies that the complete model is included and checks the APK size before publishing.
