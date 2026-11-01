# HELEN AI 3 — Gemma 4 personal Android assistant

Hebrew-first personal assistant with an on-device Gemma 4 E2B model powered by LiteRT-LM.

## AI model
- Gemma 4 E2B instruction-tuned, approximately 2.58 GB.
- The app downloads the model once on first launch into private app storage, then keeps it available offline.
- Downloads resume after interruption. Keep at least 3.1 GB free during the initial download and model preparation.
- GPU acceleration is attempted first; CPU is used if the device has no compatible OpenCL GPU runtime.
- LiteRT-LM speculative decoding is enabled for faster generation where supported.
- Model startup and inference run off the UI thread.

## Features
- Natural Hebrew chat and AI sentence completion.
- Multi-action phone controls, app launching, camera, flashlight, volume, brightness and Android settings hand-off.
- Voice input and text-to-speech.
- Reminders, timer, alarm and calendar event hand-off.
- Notes, tasks, shopping list, routines, water and expense tracking.
- Importable local text knowledge base and search.
- Backup and restore, dark/light theme, text size and speech speed settings.
- Optional accessibility actions only after explicit user opt-in.

## Build
GitHub Actions builds a debug APK. The large language model is downloaded separately on first launch so the APK remains installable without a multi-gigabyte installer.
