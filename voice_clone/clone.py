"""Voice cloning logic built on top of Coqui TTS."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"label": "English", "tts_code": "en"},
    "zh": {"label": "中文 (普通话)", "tts_code": "zh-cn"},
    "nl": {"label": "Nederlands", "tts_code": "nl"},
}

LANGUAGE_ALIASES: Dict[str, str] = {
    "zh-cn": "zh",
    "zh_cn": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "english": "en",
    "dutch": "nl",
    "nl-nl": "nl",
}


def normalise_language(language: str) -> str:
    """Convert a language alias to one of the supported canonical codes."""

    code = language.lower()
    if code in SUPPORTED_LANGUAGES:
        return code
    alias = LANGUAGE_ALIASES.get(code)
    if alias and alias in SUPPORTED_LANGUAGES:
        return alias
    raise ValueError(
        f"Unsupported language '{language}'. Choose from: {', '.join(SUPPORTED_LANGUAGES)}"
    )


REFERENCE_PROMPTS: Dict[str, list[str]] = {
    "en": [
        "The quick brown fox jumps over the lazy dog while clear bells ring in the distant valley.",
        "Technology moves quickly, but a calm voice can make even complex topics feel simple.",
        "Please read this short script so we can learn the tone, pacing, and clarity of your voice.",
    ],
    "zh": [
        "欢迎来到语音克隆演示，请保持自然语速并清晰地朗读每一个词。",
        "普通话的声调很重要，请在安静的环境里朗读这段文字。",
        "请在二十秒内介绍一下自己，保持声音稳定而自然。",
    ],
    "nl": [
        "Welkom bij deze stemklonende demo, spreek rustig en duidelijk elke zin uit.",
        "Vertel in je eigen woorden wat je vandaag van plan bent, met een natuurlijke intonatie.",
        "Lees deze korte tekst voor zodat we jouw stemkleur goed kunnen vastleggen.",
    ],
}


def get_reference_prompt(language: str = "en", randomise: bool = False) -> str:
    """Return a suggested script for recording a reference sample."""

    canonical_language = normalise_language(language)
    options = REFERENCE_PROMPTS[canonical_language]
    if randomise and len(options) > 1:
        return random.choice(options)
    return options[0]

AVAILABLE_ENGINES: Dict[str, str] = {
    "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    "xtts_v1": "tts_models/multilingual/multi-dataset/xtts_v1",
}

DEFAULT_ENGINE = "xtts_v2"


@dataclass
class VoiceProfile:
    """Represents a stored cloned voice."""

    speaker_id: str
    reference_path: Path
    description: str

    @property
    def metadata_path(self) -> Path:
        return self.reference_path.with_name("metadata.json")


class VoiceCloneService:
    """Service that records reference audio and synthesises speech in cloned voices."""

    def __init__(
        self,
        engine: str = DEFAULT_ENGINE,
        model_name: str | None = None,
        base_dir: str | Path = "voices",
        sample_rate: int = 16000,
        record_seconds: int = 20,
        use_cuda: bool | None = None,
    ) -> None:
        self.engine = engine
        self.model_name = model_name or self._resolve_engine(engine)
        self.base_dir = Path(base_dir)
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._tts: Optional[TTS] = None
        self._use_cuda = use_cuda

    @property
    def tts(self) -> TTS:
        """Lazily load the heavyweight TTS model."""

        if self._tts is None:
            self._tts = TTS(model_name=self.model_name, gpu=self._use_cuda)
        return self._tts

    def _resolve_engine(self, engine: str) -> str:
        try:
            return AVAILABLE_ENGINES[engine]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported engine '{engine}'. Choose from: {', '.join(AVAILABLE_ENGINES)}"
            ) from exc

    def set_engine(self, engine: str, model_name: str | None = None) -> None:
        """Switch to a different synthesis engine, reloading the underlying model if needed."""

        resolved_model = model_name or self._resolve_engine(engine)
        if resolved_model == self.model_name and engine == self.engine:
            return
        self.engine = engine
        self.model_name = resolved_model
        # ensure the next synthesis uses the newly selected model
        self._tts = None

    # ------------------------------------------------------------------
    # Voice management helpers
    # ------------------------------------------------------------------
    def _voice_dir(self, speaker_id: str) -> Path:
        return self.base_dir / speaker_id

    def list_voices(self) -> Iterable[VoiceProfile]:
        for path in self.base_dir.iterdir():
            reference = path / "reference.wav"
            if reference.exists():
                description = ""
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    description = metadata.get("description", "")
                yield VoiceProfile(path.name, reference, description)

    def _write_metadata(self, voice: VoiceProfile) -> None:
        metadata = {"speaker_id": voice.speaker_id, "description": voice.description}
        voice.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    def create_voice_profile(
        self,
        speaker_id: str,
        reference_wav: Path,
        description: str = "",
    ) -> VoiceProfile:
        """Store a reference recording for a speaker."""

        destination = self._voice_dir(speaker_id)
        destination.mkdir(parents=True, exist_ok=True)
        stored_reference = destination / "reference.wav"
        shutil.copy2(reference_wav, stored_reference)
        profile = VoiceProfile(speaker_id=speaker_id, reference_path=stored_reference, description=description)
        self._write_metadata(profile)
        return profile

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_reference(
        self,
        speaker_id: str,
        description: str = "",
        language: str = "en",
        scripted: bool = True,
        prompt_text: str | None = None,
        random_prompt: bool = False,
    ) -> VoiceProfile:
        """Record a reference sample from the microphone.

        Parameters
        ----------
        speaker_id:
            Identifier used to store the recorded sample.
        description:
            Optional notes saved alongside the profile.
        language:
            Language the speaker will use for the prompt/freeform speech.
        scripted:
            When True, display a suggested script to read. When False, instruct the
            speaker to talk naturally.
        prompt_text:
            Custom text to display instead of the built-in prompts.
        random_prompt:
            If True, choose one of the built-in prompts at random for additional variety.
        """

        directory = self._voice_dir(speaker_id)
        directory.mkdir(parents=True, exist_ok=True)
        reference_file = directory / "reference.wav"

        canonical_language = normalise_language(language)
        if scripted:
            script = prompt_text or get_reference_prompt(canonical_language, randomise=random_prompt)
            print(
                (
                    "Recording reference for speaker '{speaker}'. Please read the script aloud clearly.\n"
                    "Suggested text ({language}):\n{script}\n"
                    "Recording will last {seconds} seconds..."
                ).format(
                    speaker=speaker_id,
                    language=canonical_language,
                    script=script,
                    seconds=self.record_seconds,
                )
            )
        else:
            print(
                "Recording reference for speaker '%s'. Speak naturally for %s seconds.\n"
                "Try to include a mix of tones, pacing, and intonation so the model learns your style."
                % (speaker_id, self.record_seconds)
            )

        audio = sd.rec(int(self.sample_rate * self.record_seconds), samplerate=self.sample_rate, channels=1)
        sd.wait()
        sf.write(reference_file, audio, self.sample_rate)

        profile = VoiceProfile(speaker_id=speaker_id, reference_path=reference_file, description=description)
        self._write_metadata(profile)
        return profile

    def save_uploaded_reference(self, speaker_id: str, audio_path: Path, description: str = "") -> VoiceProfile:
        """Persist a reference sample provided by the GUI."""

        return self.create_voice_profile(speaker_id, audio_path, description)

    # ------------------------------------------------------------------
    # Synthesis helpers
    # ------------------------------------------------------------------
    def synthesize_to_file(
        self,
        speaker_id: str,
        text: str,
        language: str = "en",
        file_path: Optional[Path] = None,
    ) -> Path:
        """Generate speech for the given text using the cloned voice."""

        canonical_language = normalise_language(language)
        tts_language_code = SUPPORTED_LANGUAGES[canonical_language]["tts_code"]

        voice_dir = self._voice_dir(speaker_id)
        reference = voice_dir / "reference.wav"
        if not reference.exists():
            raise FileNotFoundError(f"No reference audio found for speaker '{speaker_id}'.")

        if file_path is None:
            file_path = voice_dir / f"output_{abs(hash(text))}.wav"

        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(reference),
            language=tts_language_code,
            file_path=str(file_path),
        )
        return file_path

    def synthesize(self, speaker_id: str, text: str, language: str = "en") -> tuple[int, np.ndarray]:
        """Generate speech and return it as raw audio."""

        file_path = self.synthesize_to_file(speaker_id, text, language)
        data, rate = sf.read(file_path)
        return rate, data

__all__ = [
    "VoiceCloneService",
    "SUPPORTED_LANGUAGES",
    "LANGUAGE_ALIASES",
    "AVAILABLE_ENGINES",
    "DEFAULT_ENGINE",
    "VoiceProfile",
    "normalise_language",
    "get_reference_prompt",
    "REFERENCE_PROMPTS",
]
