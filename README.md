# MyVoiceClone

MyVoiceClone is a small demo project that records a short reference clip from a user and then clones their voice using [Coqui TTS](https://github.com/coqui-ai/TTS). The cloned voice can speak English, Mandarin Chinese, or Dutch from a single reference.

## Features

- 📼 Record or upload a reference audio sample (CLI or GUI) with guided scripts or freeform speech.
- 🗣️ Clone the stored voice and synthesise new speech in English (`en`), Chinese (`zh`), or Dutch (`nl`).
- 🖥️ Launch a Gradio web UI for an end-to-end experience.
- 💾 Persist multiple speaker profiles for re-use.
- 🎛️ Switch between supported Coqui engines with XTTS v2 selected by default.

## Getting started

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Optional) Confirm that your microphone works with the `sounddevice` package.

3. Use the CLI or GUI described below to record a voice and synthesise speech.

> **Note**
> The XTTS model is large (~1.5 GB). The first run may take a few minutes while it downloads.

## Command line interface

The CLI wraps the `VoiceCloneService` class located in `voice_clone/clone.py`.

### Record a reference sample

```bash
python app.py record alice --description "Warm alto"
```

By default the recorder shows a short script tailored to the selected language and captures ~20 seconds of audio. Use `--freeform` if you prefer to speak naturally, `--random-prompt` for a different script, or `--prompt-text` to supply your own passage. The sample is saved under `voices/alice/reference.wav`.

### List stored voices

```bash
python app.py list
```

### Synthesise speech

```bash
python app.py speak alice "你好，欢迎来到语音克隆演示" --language zh
```

By default the output file is created under the speaker directory. Use `--output` to select a custom path.

To try a different engine (e.g. XTTS v1) supply the `--engine` flag when launching the CLI or GUI:

```bash
python app.py gui --engine xtts_v1
```

### Launch the GUI

```bash
python app.py gui
```

The GUI runs locally (default port 7860). It allows you to record or upload a reference sample, choose between a scripted prompt or free speech, enter text in any supported language, and generate speech. The interface stores the reference sample for later use, so you only need to record once per speaker.

### How much audio do I need?

- **Duration** – 20–30 seconds of clear, well-paced speech is enough for XTTS v2 to capture the voice timbre. Longer samples can help if the voice has unusual characteristics.
- **Scripted vs. random speech** – Consistent coverage of phonemes improves quality, so a scripted prompt is recommended for best results. However, spontaneous speech also works; just keep the recording clean and vary your intonation.
- **Environment** – Record in a quiet room with minimal background noise, avoid clipping, and speak at a steady volume.

Both the CLI and GUI provide built-in prompts and instructions so you can choose whichever style fits your workflow.

## Architecture overview

- `voice_clone/clone.py` contains the core logic for recording, storing, and synthesising voices.
- `app.py` provides both the CLI and the Gradio GUI that leverage the service.
- `requirements.txt` lists the Python dependencies.

## Limitations & future work

- Recording requires a functional microphone and may need input device selection on some systems.
- The XTTS model performs best with clear, clean reference recordings. Noisy inputs can reduce quality.
- Additional languages supported by XTTS can be exposed by updating `SUPPORTED_LANGUAGES`.
