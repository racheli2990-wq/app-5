# HELEN AI 7.0 — Qwen3 1.7B INT4 Offline Assistant

HELEN is a Hebrew-first Android personal assistant powered by **Qwen3-1.7B INT4** through Google's LiteRT-LM Android runtime.

## Built-in AI model
- Model: `litert-community/Qwen3-1.7B`
- Bundled LiteRT-LM artifact: `Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm`
- Approximate model size: **932 MB**. GitHub Actions packages it inside the APK; the app does not download a model after installation.
- On first launch the app copies the included asset into private app storage because LiteRT-LM loads a filesystem path.
- GPU is attempted first. This INT4 artifact is GPU-optimized.
- If GPU initialization fails, CPU fallback is attempted. Community measurements on a Samsung Galaxy S26 report about 40.5–40.7 generated tokens/second on GPU and 8.0–8.3 tokens/second on CPU; peak CPU RSS was about 2.5 GB. These are device-specific figures, not guarantees. Devices with limited free RAM may not be suitable for CPU fallback.
- Chat reasoning is capped at 160 tokens; sentence completion uses thinking disabled for speed.

## Features
- Hebrew-first local AI chat and sentence completion
- Quick prompts for professional writing, text summarization, translation and task planning
- Copy the latest assistant answer, share a conversation and search recent local history
- Start a new conversation and clear model context
- Phone controls and app launching
- Accessibility-assisted tap, type, scroll, read screen and screenshot (after explicit Android opt-in)
- Reminders, alarms, timers, notes, tasks, shopping, expenses, water tracking and personal routines
- Local text knowledge import and search
- Dark/light theme, text scaling, speech rate and model storage management

## Privacy and offline operation
Model inference and conversation history stay on the device. Internet may still be used by optional user-requested web or transit search features.

## License and provenance
The upstream LiteRT Community model repository declares Apache-2.0. Refer to its model card for exact details and licensing.

## Build
GitHub Actions downloads the public LiteRT-LM model at build time, packages it into `app/src/main/assets`, builds the APK, verifies the model is included and enforces a GitHub Releases-friendly APK size limit.
