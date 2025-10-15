"""Voice cloning demo application with CLI and Gradio GUI."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, Optional

import gradio as gr
import numpy as np
import soundfile as sf

from voice_clone import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    LANGUAGE_ALIASES,
    SUPPORTED_LANGUAGES,
    VoiceCloneService,
)


LANGUAGE_LABELS: Dict[str, str] = {data["label"]: code for code, data in SUPPORTED_LANGUAGES.items()}
DEFAULT_LANGUAGE_LABEL = SUPPORTED_LANGUAGES["en"]["label"]

CLI_LANGUAGE_CHOICES = sorted({*SUPPORTED_LANGUAGES.keys(), *LANGUAGE_ALIASES.keys()})

ENGINE_DISPLAY_NAMES: Dict[str, str] = {
    "xtts_v2": "Coqui XTTS v2",
    "xtts_v1": "Coqui XTTS v1",
}
ENGINE_LABELS: Dict[str, str] = {
    ENGINE_DISPLAY_NAMES.get(engine, engine): engine for engine in AVAILABLE_ENGINES
}
DEFAULT_ENGINE_LABEL = ENGINE_DISPLAY_NAMES.get(DEFAULT_ENGINE, DEFAULT_ENGINE)


def _save_gradio_audio(audio: Optional[tuple[int, np.ndarray]]) -> Optional[Path]:
    if audio is None:
        return None
    sample_rate, data = audio
    temp_file = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(temp_file, data, sample_rate)
    return temp_file


def launch_gui(service: VoiceCloneService) -> None:
    """Launch a Gradio based GUI for recording and synthesising voices."""

    def generate(text: str, language_label: str, engine_label: str, speaker_id: str, reference_audio, description: str):
        if not speaker_id:
            raise gr.Error("Please enter a speaker id (e.g. your name).")
        try:
            language = LANGUAGE_LABELS[language_label]
        except KeyError as exc:
            raise gr.Error("Unsupported language selection.") from exc
        try:
            engine = ENGINE_LABELS[engine_label]
        except KeyError as exc:
            raise gr.Error("Unsupported engine selection.") from exc
        service.set_engine(engine)
        reference_path = _save_gradio_audio(reference_audio)
        if reference_path is not None:
            service.save_uploaded_reference(speaker_id, reference_path, description)
        rate, data = service.synthesize(speaker_id, text, language)
        return rate, data

    with gr.Blocks(title="MyVoiceClone") as demo:
        gr.Markdown(
            """
            # MyVoiceClone
            1. Enter a speaker id (e.g. your name).
            2. Record or upload a short reference clip by reading a few sentences.
            3. Provide text and choose the output language (English, 中文, or Nederlands).
            4. Click **Generate Speech** to hear the cloned voice.
            """
        )
        with gr.Row():
            speaker = gr.Textbox(label="Speaker ID", placeholder="jane_doe")
            language = gr.Dropdown(
                choices=list(LANGUAGE_LABELS.keys()), value=DEFAULT_LANGUAGE_LABEL, label="Language"
            )
            engine = gr.Dropdown(
                choices=list(ENGINE_LABELS.keys()), value=DEFAULT_ENGINE_LABEL, label="TTS Engine"
            )
        reference = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Reference sample")
        description = gr.Textbox(label="Description", placeholder="Optional notes about the voice", lines=1)
        text = gr.Textbox(label="Text to speak", placeholder="请输入文本 / Voer tekst in / Enter text", lines=4)
        generate_button = gr.Button("Generate Speech")
        output_audio = gr.Audio(label="Synthesised audio", type="numpy")

        generate_button.click(
            generate,
            inputs=[text, language, engine, speaker, reference, description],
            outputs=output_audio,
        )

    demo.launch()


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Voice cloning demo")
    parser.add_argument("--base-dir", default="voices", help="Directory to store cloned voices")
    parser.add_argument(
        "--engine",
        choices=list(AVAILABLE_ENGINES.keys()),
        default=DEFAULT_ENGINE,
        help="Synthesis engine to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional custom model path. Overrides the engine choice when provided.",
    )
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration when available")

    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record a new voice reference from the microphone")
    record_parser.add_argument("speaker", help="Identifier for the speaker (e.g. alice)")
    record_parser.add_argument("--description", default="", help="Free form description for the voice")

    list_parser = subparsers.add_parser("list", help="List stored voice profiles")

    speak_parser = subparsers.add_parser("speak", help="Synthesize speech for an existing voice")
    speak_parser.add_argument("speaker", help="Speaker identifier")
    speak_parser.add_argument("text", help="Text to speak")
    speak_parser.add_argument("--language", default="en", choices=CLI_LANGUAGE_CHOICES)
    speak_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output wav file. If omitted an auto generated path within the speaker folder is used.",
    )

    subparsers.add_parser("gui", help="Launch the Gradio web interface")

    args = parser.parse_args(argv)

    service = VoiceCloneService(
        engine=args.engine,
        model_name=args.model,
        base_dir=args.base_dir,
        use_cuda=args.cuda,
    )

    if args.command == "record":
        profile = service.record_reference(args.speaker, description=args.description)
        print(f"Recorded reference stored at {profile.reference_path}")
    elif args.command == "list":
        for voice in service.list_voices():
            lang_info = ", ".join(
                f"{code} ({SUPPORTED_LANGUAGES[code]['label']})" for code in SUPPORTED_LANGUAGES
            )
            print(
                "Speaker: {speaker}\n  Description: {description}\n  Languages: {languages}\n  Reference: {reference}\n".format(
                    speaker=voice.speaker_id,
                    description=voice.description,
                    languages=lang_info,
                    reference=voice.reference_path,
                )
            )
    elif args.command == "speak":
        output = service.synthesize_to_file(args.speaker, args.text, args.language, args.output)
        print(f"Synthesised speech saved to {output}")
    elif args.command == "gui":
        launch_gui(service)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    cli()
