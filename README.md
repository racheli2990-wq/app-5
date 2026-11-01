# HELEN AI 4.0 — Personal Android Assistant

A Hebrew-first Android assistant powered by **Gemma 4 E2B** using Google's LiteRT-LM runtime.

## AI model and offline use
- The app installs as a compact Android APK, then downloads the approximately **2.58 GB** Gemma 4 E2B model once on first launch.
- The model is stored in the app's private storage. After the first download, chat, tool interpretation and smart autocomplete run locally without a network connection.
- The first model download requires Wi-Fi or another internet connection and several GB of free storage.
- The runtime tries GPU acceleration with Multi-Token Prediction (MTP), then falls back to CPU if the device does not support the required GPU backend. MTP is disabled on CPU for the E2B free-form chat path.
- Model performance and memory usage depend on the Android device. The published Google benchmark uses a Samsung S26 Ultra; it is not a promise for every phone.

## Features
- Natural Hebrew chat and AI-generated sentence completion
- Multi-command interpretation for phone settings and app launch
- Accessibility-assisted tap, type, scroll, read screen and screenshot (after explicit Android opt-in)
- Reminders, alarms, timers, notes, tasks, shopping list, expenses and water tracking
- Personal routines, local text knowledge import, calendar handoff
- Dark/light theme, text scaling, speech rate, chat history restore and model storage management

## Important Android limits
Wi-Fi/Bluetooth radios are handled through Android settings; phone calls and WhatsApp messages require the user's confirmation. Speech recognition availability depends on the device's recognition service.

## Build
Use GitHub Actions workflow **Build Helen AI 4 APK**. It verifies the model endpoint and builds a debug APK.
