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
   # macOS / Linux (bash)
   python -m venv .venv
   py -3.11 -m venv .venv
   source .venv/bin/activate

   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1     
   pip install -r requirements.txt
   ```

   Windows (PowerShell):

   ```powershell
   # Use 64-bit Python 3.11 for best compatibility
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   # If scripts are blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

   # Upgrade build tools
   python -m pip install -U pip setuptools wheel

   # Install PyTorch first (choose one):
   # CPU only:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   # NVIDIA GPU (CUDA 12.1):
   # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Then install the rest
   pip install -r requirements.txt
   ```

   Notes:
   - Recommended Python: 3.10 or 3.11 (64‑bit). TTS wheels may be unavailable on 3.12+ for Windows.
   - Render deploys default to Python 3.13, which currently lacks compatible TTS wheels. Add a `runtime.txt` containing `3.10.14` (already included in this repo) so Render installs Python 3.10 during builds.
   - If you still see “No matching distribution found for TTS”, try: `pip install "TTS==0.15.3"` before installing the rest.

2. (Optional) Confirm that your microphone works with the `sounddevice` package.

3. Use the CLI or GUI described below to record a voice and synthesise speech.

> **Note**
> The XTTS model is large (~1.5 GB). The first run may take a few minutes while it downloads.

Generated audio is stored in the `voices/` directory (and any `.wav` exports you create).
These artefacts are ignored by git via the provided `.gitignore` so that cloned profiles do
not accidentally get committed.
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

> 💡 Tip: running `python app.py` without a sub-command now launches the GUI automatically. This is handy for platforms such as Render that expect the process to start a web server on the assigned `$PORT`.

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

## Deploying to Vercel (Container)

Vercel’s serverless Python functions aren’t suited for long‑lived Gradio servers. Deploy this app on Vercel using the included Dockerfile so it can bind to the platform’s `PORT`.

- The repository includes a `Dockerfile` that runs `python app.py` and listens on `0.0.0.0:$PORT`.
- Create a new Vercel project from this repository; Vercel will auto‑detect the Dockerfile and build a container.
- No extra configuration is required. The app logs will show the selected host/port when it boots.

Notes:
- Microphone recording inside containers may require PortAudio. Uncomment the `libportaudio2` install lines in `Dockerfile` if you need mic recording in the container. Uploading reference `.wav` files works without it.
- Large ML dependencies can exceed free‑tier limits. If builds time out or the image is too big, consider prebuilding and pushing to a registry, upgrading the plan, or using a platform geared for larger images.
