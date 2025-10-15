"""Voice cloning demo application with CLI and Gradio GUI."""

from __future__ import annotations

import argparse
import json
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
    get_reference_prompt,
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

REFERENCE_STYLE_LABELS: Dict[str, str] = {
    "Scripted reference (recommended)": "scripted",
    "Free speech sample": "free",
}

DEFAULT_REFERENCE_STYLE_LABEL = "Scripted reference (recommended)"


def _save_gradio_audio(audio: Optional[tuple[int, np.ndarray]]) -> Optional[Path]:
    if audio is None:
        return None
    sample_rate, data = audio
    temp_file = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(temp_file, data, sample_rate)
    return temp_file


def launch_gui(service: VoiceCloneService) -> None:
    """Launch a Gradio based GUI for recording and synthesising voices."""

    def _set_last_language(speaker_id: str, language_code: str) -> None:
        try:
            meta_path = service.base_dir / speaker_id / "metadata.json"
            meta: Dict[str, object] = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["speaker_id"] = speaker_id
            if "description" not in meta:
                meta["description"] = ""
            meta["last_language"] = language_code
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Non-fatal
            pass

    def _get_last_language_label(speaker_id: str) -> Optional[str]:
        try:
            meta_path = service.base_dir / speaker_id / "metadata.json"
            if not meta_path.exists():
                return None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            code = meta.get("last_language")
            if isinstance(code, str) and code in SUPPORTED_LANGUAGES:
                return SUPPORTED_LANGUAGES[code]["label"]
        except Exception:
            pass
        return None

    def generate(
        text: str,
        language_label: str,
        engine_label: str,
        speaker_id: str,
        reference_audio,
        description: str,
    ):
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
        rate, data = service.synthesize(speaker_id, text, language, description=description)
        _set_last_language(speaker_id, language)
        return rate, data

    with gr.Blocks(title="MyVoiceClone") as demo:
        gr.Markdown(
            """
            # MyVoiceClone
            1. Enter a speaker id (e.g. your name).
            2. Record or upload a ~20-30 second reference clip using the scripted prompt or free speech.
            3. Provide text and choose the output language (English, 中文, or Nederlands).
            4. Click **Generate Speech** to hear the cloned voice.
            """
        )
        with gr.Row():
            existing_speakers = gr.Dropdown(
                choices=[v.speaker_id for v in service.list_voices()],
                label="Existing Speakers",
            )
            refresh_speakers = gr.Button("Refresh Speakers")
            speaker = gr.Textbox(label="Speaker ID", placeholder="jane_doe")
        with gr.Row():
            language = gr.Dropdown(
                choices=list(LANGUAGE_LABELS.keys()), value=DEFAULT_LANGUAGE_LABEL, label="Language"
            )
            engine = gr.Dropdown(
                choices=list(ENGINE_LABELS.keys()), value=DEFAULT_ENGINE_LABEL, label="TTS Engine"
            )
        reference = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Reference sample")
        reference_style = gr.Radio(
            choices=list(REFERENCE_STYLE_LABELS.keys()),
            value=DEFAULT_REFERENCE_STYLE_LABEL,
            label="Reference style",
        )
        reference_prompt = gr.Textbox(
            label="Suggested script",
            value=get_reference_prompt("en"),
            interactive=False,
            lines=3,
            show_copy_button=True,
        )
        random_prompt_button = gr.Button("New scripted prompt")
        reference_instructions = gr.Markdown(
            "Read the script aloud for about 20-30 seconds. Switch to Free speech sample if you prefer to improvise.",
        )
        save_status = gr.Markdown(visible=False)
        reference_status = gr.Markdown(visible=False)
        description = gr.Textbox(label="Description", placeholder="Optional notes about the voice", lines=1)
        text = gr.Textbox(label="Text to speak", placeholder="请输入文本 / Voer tekst in / Enter text", lines=4)
        with gr.Row():
            save_button = gr.Button("Save Reference Only")
            generate_button = gr.Button("Generate Speech")
        output_audio = gr.Audio(label="Synthesised audio", type="numpy")

        generate_button.click(
            generate,
            inputs=[text, language, engine, speaker, reference, description],
            outputs=output_audio,
        )

        def check_reference_status(speaker_id: str):
            if not speaker_id:
                return gr.update(visible=False)
            ref = service.base_dir / speaker_id / "reference.wav"
            if ref.exists():
                return gr.update(
                    visible=True,
                    value=f"Found existing reference for '{speaker_id}' at '{ref}'. You can generate without re-recording.",
                )
            return gr.update(visible=False)

        speaker.change(check_reference_status, inputs=[speaker], outputs=[reference_status])

        def choose_existing(sel: str):
            if not sel:
                return gr.update(), gr.update(), gr.update(visible=False)
            # Update language if metadata has last_language
            lang_label = _get_last_language_label(sel)
            lang_update = gr.update(value=lang_label) if lang_label else gr.update()
            return gr.update(value=sel), lang_update, check_reference_status(sel)

        existing_speakers.change(
            choose_existing,
            inputs=[existing_speakers],
            outputs=[speaker, language, reference_status],
        )

        def refresh_existing():
            return gr.update(choices=[v.speaker_id for v in service.list_voices()])

        refresh_speakers.click(refresh_existing, inputs=None, outputs=existing_speakers)

        def _build_prompt(language_label: str, style_label: str, randomise: bool = False) -> tuple[str, str]:
            language_code = LANGUAGE_LABELS[language_label]
            style = REFERENCE_STYLE_LABELS[style_label]
            if style == "scripted":
                script = get_reference_prompt(language_code, randomise=randomise)
                instruction = (
                    "Read the script aloud for about 20-30 seconds. A clear recording with minimal background noise gives the best results."
                )
            else:
                script = (
                    "Speak naturally for 20-30 seconds about any topic—describe your day, narrate a story, or read any passage you like."
                )
                instruction = (
                    "Keep your voice steady and expressive. Mixing different tones and pacing helps the model capture your style."
                )
            return script, instruction

        def update_reference_prompt(language_label: str, style_label: str) -> tuple[str, str]:
            return _build_prompt(language_label, style_label)

        def randomise_prompt(language_label: str, style_label: str) -> tuple[str, str]:
            return _build_prompt(language_label, style_label, randomise=True)

        language.change(
            update_reference_prompt,
            inputs=[language, reference_style],
            outputs=[reference_prompt, reference_instructions],
        )
        reference_style.change(
            update_reference_prompt,
            inputs=[language, reference_style],
            outputs=[reference_prompt, reference_instructions],
        )
        random_prompt_button.click(
            randomise_prompt,
            inputs=[language, reference_style],
            outputs=[reference_prompt, reference_instructions],
        )

        def save_only(speaker_id: str, reference_audio, description: str, language_label: str):
            if not speaker_id:
                raise gr.Error("Please enter a speaker id (e.g. your name).")
            reference_path = _save_gradio_audio(reference_audio)
            if reference_path is None:
                raise gr.Error("Please record or upload a reference sample to save.")
            service.save_uploaded_reference(speaker_id, reference_path, description)
            try:
                lang_code = LANGUAGE_LABELS[language_label]
            except KeyError:
                lang_code = "en"
            _set_last_language(speaker_id, lang_code)
            return gr.update(
                visible=True,
                value=f"Saved reference for '{speaker_id}'. You can now generate without re-recording.",
            )

        save_button.click(
            save_only,
            inputs=[speaker, reference, description, language],
            outputs=save_status,
        )

    # Launch the app; disable analytics when supported, otherwise fall back.
    try:
        demo.launch(analytics_enabled=False, server_name="127.0.0.1")
    except TypeError:
        try:
            demo.launch(server_name="127.0.0.1")
        except TypeError:
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
    record_parser.add_argument(
        "--language",
        default="en",
        choices=CLI_LANGUAGE_CHOICES,
        help="Language of the reference script or freeform speech",
    )
    record_parser.add_argument(
        "--freeform",
        action="store_true",
        help="Record a spontaneous sample instead of reading the scripted prompt",
    )
    record_parser.add_argument(
        "--random-prompt",
        action="store_true",
        help="Pick a different scripted prompt at random",
    )
    record_parser.add_argument(
        "--prompt-text",
        default=None,
        help="Custom text to display when recording. Overrides the preset prompts.",
    )

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
    speak_parser.add_argument(
        "--description",
        default=None,
        help="Optional description used to build the output filename when --output is not provided.",
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
        profile = service.record_reference(
            args.speaker,
            description=args.description,
            language=args.language,
            scripted=not args.freeform,
            prompt_text=args.prompt_text,
            random_prompt=args.random_prompt,
        )
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
        output = service.synthesize_to_file(
            args.speaker,
            args.text,
            args.language,
            args.output,
            description=args.description,
        )
        print(f"Synthesised speech saved to {output}")
    elif args.command == "gui":
        launch_gui(service)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    cli()
