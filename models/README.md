# Models Directory

This directory contains the machine learning models required for the voice assistant.

## Required Models

### Wake Word Detection
- `alexa_v0.1.onnx` - Wake word detection model for "Alexa"
- `embedding_model.onnx` - Embedding model for wake word processing
- `melspectrogram.onnx` - Audio feature extraction model

### Other Available Wake Words (optional)
- `hey_jarvis_v0.1.onnx`
- `hey_marvin_v0.1.onnx`
- `hey_mycroft_v0.1.onnx`

### Speech Models (to be added)
- Whisper models for STT
- Kokoro models for TTS

## Note
Model files (*.onnx) are not included in the repository due to their large size.
You need to download them separately and place them in this directory.
