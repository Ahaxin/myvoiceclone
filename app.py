"""Voice cloning demo application with CLI and Gradio GUI."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from ipaddress import ip_address
from pathlib import Path
from typing import Dict, Optional

import gradio as gr
import numpy as np
import soundfile as sf

from voice_clone import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    LANGUAGE_ALIASES,
    MicrophoneUnavailableError,
    REFERENCE_PROMPTS,
    SUPPORTED_LANGUAGES,
    VoiceCloneService,
    get_reference_prompt,
    normalise_language,
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

    def _build_speaker_library() -> Dict[str, Dict[str, object]]:
        library: Dict[str, Dict[str, object]] = {}
        for profile in service.list_voices():
            library[profile.speaker_id] = {
                "description": profile.description,
                "last_language": profile.last_language,
                "reference_path": str(profile.reference_path),
            }
        return library

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

    def generate(
        text: str,
        language_label: str,
        engine_label: str,
        speaker_id: str,
        reference_audio,
        description: str,
    ):
        speaker_id = speaker_id.strip()
        if not speaker_id:
            raise gr.Error("Please enter a speaker id (e.g. your name).")
        if not text or not text.strip():
            raise gr.Error("Please provide text to synthesise.")
        text = text.strip()
        description = description.strip() if isinstance(description, str) else ""
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
            service.save_uploaded_reference(
                speaker_id,
                reference_path,
                description,
                language=language,
            )
            try:
                reference_path.unlink()
            except OSError:
                pass
        try:
            rate, data = service.synthesize(
                speaker_id, text, language, description=description
            )
        except FileNotFoundError as exc:
            raise gr.Error(
                "No reference found for this speaker yet. Please record or upload a sample first."
            ) from exc
        _set_last_language(speaker_id, language)
        return rate, data

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

    def save_only(speaker_id: str, reference_audio, description: str, language_label: str):
        speaker_id = speaker_id.strip()
        if not speaker_id:
            raise gr.Error("Please enter a speaker id (e.g. your name).")
        reference_path = _save_gradio_audio(reference_audio)
        if reference_path is None:
            raise gr.Error("Please record or upload a reference sample to save.")
        description = description.strip() if isinstance(description, str) else ""
        try:
            lang_code = LANGUAGE_LABELS[language_label]
        except KeyError:
            lang_code = "en"
        service.save_uploaded_reference(
            speaker_id,
            reference_path,
            description,
            language=lang_code,
        )
        try:
            reference_path.unlink()
        except OSError:
            pass
        _set_last_language(speaker_id, lang_code)
        return gr.update(
            visible=True,
            value=f"Saved reference for '{speaker_id}'. You can now generate without re-recording.",
        )

    def refresh_existing(selected: Optional[str] = None):
        library = _build_speaker_library()
        choices = sorted(library.keys())
        value = selected if selected in library else None
        return gr.update(choices=choices, value=value), library

    def choose_existing(sel: str, library: Dict[str, Dict[str, object]]):
        if not sel:
            return (
                gr.update(),
                gr.update(),
                gr.update(value=""),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        metadata = library.get(sel, {})
        lang_label = None
        lang_code = metadata.get("last_language")
        if isinstance(lang_code, str) and lang_code in SUPPORTED_LANGUAGES:
            lang_label = SUPPORTED_LANGUAGES[lang_code]["label"]
        lang_update = gr.update(value=lang_label) if lang_label else gr.update()
        description_update = gr.update(value=str(metadata.get("description", "")))

        summary_parts = []
        if lang_label and lang_code:
            summary_parts.append(f"**Last language used:** {lang_label} ({lang_code})")
        reference_path = metadata.get("reference_path")
        if isinstance(reference_path, str) and reference_path:
            summary_parts.append(f"**Reference file:** `{reference_path}`")
        summary_parts.append("You can update the description or record a new sample at any time.")
        summary_text = "\n\n".join(summary_parts)

        return (
            gr.update(value=sel),
            lang_update,
            description_update,
            check_reference_status(sel),
            gr.update(visible=True, value=summary_text),
        )

    initial_library = _build_speaker_library()

    with gr.Blocks(
        title="MyVoiceClone",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="cyan"),
    ) as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">
            <h1>MyVoiceClone</h1>
            <p>Create lifelike voice clones in minutes. Follow the guided steps below.</p>
            </div>
            <ol>
              <li><strong>Choose or name a speaker</strong> – each voice profile is stored for reuse.</li>
              <li><strong>Record or upload a reference</strong> – a clear 20–30 second clip works best.</li>
              <li><strong>Enter the text and language</strong> you want to hear in the cloned voice.</li>
            </ol>
            """,
            elem_id="app-header",
        )

        speaker_library = gr.State(initial_library)

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=520):
                gr.Markdown("### 1 · Speaker profile")
                with gr.Row():
                    existing_speakers = gr.Dropdown(
                        choices=sorted(initial_library.keys()),
                        label="Select an existing speaker",
                        value=None,
                    )
                    refresh_speakers = gr.Button("Refresh", variant="secondary")
                speaker = gr.Textbox(
                    label="Speaker ID",
                    placeholder="e.g. jane_doe",
                    info="Use letters, numbers, or underscores to identify the voice profile.",
                )
                description = gr.Textbox(
                    label="Description",
                    placeholder="Optional notes – accent, style, project name…",
                    lines=2,
                )
                speaker_summary = gr.Markdown(visible=False)
                reference_status = gr.Markdown(visible=False)

                gr.Markdown("### 2 · Reference recording")
                with gr.Row():
                    language = gr.Dropdown(
                        choices=list(LANGUAGE_LABELS.keys()),
                        value=DEFAULT_LANGUAGE_LABEL,
                        label="Voice language",
                    )
                    engine = gr.Dropdown(
                        choices=list(ENGINE_LABELS.keys()),
                        value=DEFAULT_ENGINE_LABEL,
                        label="TTS engine",
                    )

                reference_style = gr.Radio(
                    choices=list(REFERENCE_STYLE_LABELS.keys()),
                    value=DEFAULT_REFERENCE_STYLE_LABEL,
                    label="Reference style",
                    info="Scripted prompts capture pronunciation; free speech captures natural style.",
                )
                with gr.Row():
                    random_prompt_button = gr.Button(
                        "Shuffle scripted prompt", variant="secondary"
                    )
                reference_prompt = gr.Textbox(
                    label="Suggested script",
                    value=get_reference_prompt("en"),
                    interactive=False,
                    lines=4,
                    show_copy_button=True,
                )
                reference_instructions = gr.Markdown(
                    "Read the script aloud for about 20–30 seconds. Switch to Free speech sample to improvise instead.",
                )

                reference = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Reference sample",
                    show_download_button=False,
                )
                with gr.Row():
                    save_button = gr.Button("Save reference only", variant="secondary")
                save_status = gr.Markdown(visible=False)

            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### 3 · Generate speech")
                text = gr.Textbox(
                    label="Text to speak",
                    placeholder="请输入文本 / Voer tekst in / Enter text",
                    lines=6,
                )
                generate_button = gr.Button("Generate speech", variant="primary")
                output_audio = gr.Audio(label="Synthesised audio", type="numpy")

                gr.Markdown(
                    """
                    #### Tips for natural results
                    - Record in a quiet room and keep the microphone at a steady distance.
                    - Speak with consistent pacing and energy.
                    - Re-record or update the reference whenever the voice changes tone or emotion.
                    - Try different scripted prompts to capture varied phonetics.
                    """
                )

        generate_button.click(
            generate,
            inputs=[text, language, engine, speaker, reference, description],
            outputs=output_audio,
        ).then(
            refresh_existing,
            inputs=[speaker],
            outputs=[existing_speakers, speaker_library],
        )

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

        save_button.click(
            save_only,
            inputs=[speaker, reference, description, language],
            outputs=save_status,
        ).then(
            refresh_existing,
            inputs=[speaker],
            outputs=[existing_speakers, speaker_library],
        )

        existing_speakers.change(
            choose_existing,
            inputs=[existing_speakers, speaker_library],
            outputs=[speaker, language, description, reference_status, speaker_summary],
        )

        refresh_speakers.click(
            refresh_existing,
            inputs=[existing_speakers],
            outputs=[existing_speakers, speaker_library],
        )

        speaker.change(
            lambda speaker_id: (check_reference_status(speaker_id), gr.update(visible=False)),
            inputs=[speaker],
            outputs=[reference_status, speaker_summary],
        )

    # Launch the app; disable analytics when supported, otherwise fall back.
    host_env = os.environ.get("HOST")
    running_on_render = any(
        key in os.environ
        for key in ("RENDER", "RENDER_EXTERNAL_HOSTNAME", "RENDER_SERVICE_NAME")
    )

    host: str | None = None
    if host_env:
        try:
            # Some platforms (including Render) set HOST to the public hostname, which
            # cannot be bound directly. Only honour the variable when it contains a
            # valid IP literal and fall back to 0.0.0.0 otherwise.
            ip_address(host_env)
            host = host_env
        except ValueError:
            host = None

    if host is None:
        host = "0.0.0.0" if running_on_render else "127.0.0.1"
    default_port = 7860
    port_value = os.environ.get("PORT", str(default_port))
    try:
        port = int(port_value)
    except (TypeError, ValueError):
        port = default_port

    launch_variants = [
        {"analytics_enabled": False, "server_name": host, "server_port": port},
        {"server_name": host, "server_port": port},
        {"server_port": port},
        {"server_name": host},
        {},
    ]

    print(f"Starting MyVoiceClone GUI on {host}:{port} (Render={running_on_render})")
    for kwargs in launch_variants:
        try:
            demo.launch(**kwargs)
            break
        except TypeError:
            continue


def _cli_examples() -> str:
    return textwrap.dedent(
        """
        Examples:
          python app.py gui
          python app.py record alice --language en
          python app.py speak alice "Hello there!" --language en --output hello.wav
          python app.py list
          python app.py prompts --language zh --all
        """
    ).strip()


def build_parser() -> tuple[argparse.ArgumentParser, Dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog="myvoiceclone",
        description="Interact with the MyVoiceClone toolkit via the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_cli_examples(),
    )
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

    subparsers = parser.add_subparsers(dest="command")
    command_parsers: Dict[str, argparse.ArgumentParser] = {}

    record_parser = subparsers.add_parser(
        "record",
        help="Record a new voice reference from the microphone",
        description="Record a microphone sample to create or update a speaker profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    command_parsers["record"] = record_parser

    list_parser = subparsers.add_parser(
        "list",
        help="List stored voice profiles",
        description="Show stored speakers, their descriptions, and where the reference audio lives.",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output the speaker catalogue as JSON for scripting.",
    )
    command_parsers["list"] = list_parser

    speak_parser = subparsers.add_parser(
        "speak",
        help="Synthesize speech for an existing voice",
        description="Generate speech in a stored voice and optionally save it to disk.",
    )
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
    command_parsers["speak"] = speak_parser

    gui_parser = subparsers.add_parser("gui", help="Launch the Gradio web interface")
    command_parsers["gui"] = gui_parser

    languages_parser = subparsers.add_parser(
        "languages",
        help="Show supported languages and their aliases",
        description="Display the language codes you can use with the CLI commands.",
    )
    command_parsers["languages"] = languages_parser

    prompts_parser = subparsers.add_parser(
        "prompts",
        help="Display reference scripts for a language",
        description="Preview or randomise the scripted prompts used when recording references.",
    )
    prompts_parser.add_argument(
        "--language",
        default="en",
        choices=CLI_LANGUAGE_CHOICES,
        help="Language code or alias for the prompt list.",
    )
    prompts_parser.add_argument(
        "--all",
        action="store_true",
        help="List every built-in prompt for the chosen language.",
    )
    prompts_parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random prompt instead of the default ordering.",
    )
    command_parsers["prompts"] = prompts_parser

    help_parser = subparsers.add_parser(
        "help",
        help="Show detailed help for a specific command",
        description="Display usage information for the entire tool or a sub-command.",
    )
    help_parser.add_argument("topic", nargs="?", help="Command to describe in detail")
    command_parsers["help"] = help_parser

    # Allow running the script without an explicit sub-command. When no
    # command is provided (e.g. the deploy platform invokes `python app.py`),
    # the CLI will default to launching the GUI. This keeps the more explicit
    # sub-commands available for power users while ensuring hosted
    # environments start a web server and bind to the expected port.
    subparsers.required = False
    return parser, command_parsers


def cli(argv: list[str] | None = None) -> None:
    parser, command_parsers = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        args.command = "gui"

    def _print_languages() -> None:
        alias_map: Dict[str, list[str]] = {}
        for alias, canonical in LANGUAGE_ALIASES.items():
            alias_map.setdefault(canonical, []).append(alias)
        print("Supported languages:\n")
        for code, data in SUPPORTED_LANGUAGES.items():
            aliases = ", ".join(sorted(alias_map.get(code, []))) or "(no aliases)"
            print(f"  {code:<6} {data['label']}  aliases: {aliases}")

    def _handle_prompts(language_choice: str, list_all: bool, pick_random: bool) -> None:
        canonical = normalise_language(language_choice)
        prompts = REFERENCE_PROMPTS[canonical]
        header = f"Reference prompts for {SUPPORTED_LANGUAGES[canonical]['label']} ({canonical}):"
        print(header)
        print("-" * len(header))
        if pick_random:
            print(get_reference_prompt(canonical, randomise=True))
            return
        if list_all:
            for idx, script in enumerate(prompts, start=1):
                print(f"[{idx}] {script}")
            return
        print(prompts[0])

    if args.command == "help":
        topic = getattr(args, "topic", None)
        if topic:
            sub = command_parsers.get(topic)
            if sub is None:
                available = ", ".join(sorted(k for k in command_parsers.keys() if k != "help"))
                print(f"Unknown topic '{topic}'. Available commands: {available}")
            else:
                sub.print_help()
        else:
            parser.print_help()
        return

    if args.command == "languages":
        _print_languages()
        return

    if args.command == "prompts":
        _handle_prompts(args.language, args.all, args.random)
        return

    service = VoiceCloneService(
        engine=args.engine,
        model_name=args.model,
        base_dir=args.base_dir,
        use_cuda=args.cuda,
    )

    if args.command == "record":
        try:
            profile = service.record_reference(
                args.speaker,
                description=args.description,
                language=args.language,
                scripted=not args.freeform,
                prompt_text=args.prompt_text,
                random_prompt=args.random_prompt,
            )
        except MicrophoneUnavailableError as exc:
            raise SystemExit(str(exc))
        print(f"Recorded reference stored at {profile.reference_path}")
    elif args.command == "list":
        voices = list(service.list_voices())
        if args.json:
            serialised = [
                {
                    "speaker_id": voice.speaker_id,
                    "description": voice.description,
                    "last_language": voice.last_language,
                    "reference_path": str(voice.reference_path),
                }
                for voice in voices
            ]
            print(json.dumps(serialised, indent=2, ensure_ascii=False))
            return
        if not voices:
            print("No stored voices were found. Record a reference with 'record' or upload one via the GUI.")
            return
        print(f"Found {len(voices)} speaker{'s' if len(voices) != 1 else ''}:")
        for voice in voices:
            print(f"- {voice.speaker_id}")
            if voice.description:
                print(f"    Description : {voice.description}")
            if voice.last_language and voice.last_language in SUPPORTED_LANGUAGES:
                label = SUPPORTED_LANGUAGES[voice.last_language]["label"]
                print(f"    Last language: {label} ({voice.last_language})")
            print(f"    Reference   : {voice.reference_path}")
        print(
            "\nVoices can synthesise any supported language: "
            + ", ".join(f"{code} ({data['label']})" for code, data in SUPPORTED_LANGUAGES.items())
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
