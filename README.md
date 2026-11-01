# HELEN AI 5.0 — Bundled Offline Assistant

A Hebrew-first Android personal assistant using **Qwen3-0.6B INT4 (no-think)** with Google's LiteRT-LM runtime.

## Built-in AI model
- The approximately **347 MB** Qwen3 0.6B INT4 no-think model is embedded in the APK as an application asset.
- There is **no model download after installation**. On first launch, the app copies the embedded model into its private app storage because LiteRT-LM loads it from a filesystem path.
- After installation, chat and generated sentence suggestions run locally without an internet connection.
- The app tries the GPU backend first and falls back to CPU when the device GPU backend is unavailable.
- The no-think model variant is configured for direct responses and is designed to be faster than the reasoning-enabled variant, especially on CPU.
- Model speed, memory use and Hebrew quality depend on the phone. The upstream model card reports benchmark results on specific devices; those results are not a guarantee for every Android phone. Hebrew is not one of the model card's listed primary languages, so Hebrew quality should be tested on the target device.

## Features
- Natural-language chat and AI-generated sentence completion
- Multi-command interpretation for phone settings and app launch
- Accessibility-assisted tap, type, scroll, read screen and screenshot (after explicit Android opt-in)
- Reminders, alarms, timers, notes, tasks, shopping list, expenses and water tracking
- Personal routines and local text knowledge import
- Dark/light theme, text scaling, speech rate, chat history and model storage management

## Model provenance and license
- Model: `litert-community/Qwen3-0.6B-int4`
- File: `qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm`
- Base model: `Qwen/Qwen3-0.6B`
- Model repository declares Apache-2.0. See the model card and license at Hugging Face.

## Android limits
Wi-Fi/Bluetooth radios are handled through Android settings; phone calls and WhatsApp messages require user confirmation. Speech recognition availability depends on the device's recognition service. Some screen-control actions require the user's explicit accessibility opt-in.

## Build
GitHub Actions downloads the public model at build time, packages it into `app/src/main/assets`, builds the debug APK and checks that the model file is actually present inside the APK.
