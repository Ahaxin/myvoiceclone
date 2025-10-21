"""Voice cloning logic built on top of Coqui TTS."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import soundfile as sf
from TTS.api import TTS
from datetime import datetime
import re

try:
    import sounddevice as sd
    _SOUNDDEVICE_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError) as exc:  # pragma: no cover - import guard
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_IMPORT_ERROR = exc

_HAS_SOUNDDEVICE = sd is not None

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
        (
            "Hello there, I'm speaking at a relaxed pace so you can hear the gentle rise and fall of my"
            " everyday tone. I start by sharing how my mornings unfold, from the first sip of coffee to a"
            " quiet walk outside where the air feels calm and bright. As I move into describing the work I"
            " love and the teammates I collaborate with, notice how my voice keeps a steady rhythm, how"
            " certain words stretch slightly for warmth, and how every pause gives space without breaking"
            " the flow. I finish by talking about the creative projects that excite me, the challenges that"
            " spark curiosity, and the laughter that lightens serious moments, ending with a friendly"
            " cadence you can almost see as a smile."
        )
    ],
    "zh": [
        (
            "大家好，我会用自然平稳的语速介绍自己的日常节奏，从清晨舒展身体的晨练，到冲泡一杯暖暖"
            " 的咖啡，再到坐下来规划今天的任务。接着我谈到与团队合作时的角色，如何耐心倾听、温柔回"
            " 应，并在需要强调重点时让语调轻轻升高又缓缓落下，让人感到安心。然后我分享最近阅读的书"
            " 和喜欢的音乐，还描述与朋友聚会时愉快的笑声。整个叙述保持清晰、柔和而富有表情，最后以"
            " 略带微笑的尾音收束，让听者感到亲切自然。"
        )
    ],
    "nl": [
        (
            "Hallo, ik spreek op een rustig tempo terwijl ik vertel hoe mijn ochtend begint, van het zetten"
            " van koffie tot het moment waarop ik de frisse buitenlucht inadem tijdens een korte wandeling."
            " Daarna neem ik je mee naar mijn werkdag, waar ik met collega's ideeën uitwissel, met aandacht"
            " luister en met vertrouwen mijn eigen inzichten deel. Je hoort hoe mijn stem iets hoger klinkt"
            " wanneer ik enthousiasme toon en weer terugzakt naar een warme, gelijkmatige toon om het verhaal"
            " vloeiend te houden. Tot slot beschrijf ik mijn vrije tijd: wandelen in het park, nieuwe"
            " recepten uitproberen en samen lachen om kleine grappen, allemaal uitgesproken met duidelijke"
            " maar vriendelijke articulatie die mijn persoonlijke stijl laat horen."
        )
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


class MicrophoneUnavailableError(RuntimeError):
    """Raised when microphone recording is not possible in the current environment."""


@dataclass
class VoiceProfile:
    """Represents a stored cloned voice."""

    speaker_id: str
    reference_path: Path
    description: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def metadata_path(self) -> Path:
        return self.reference_path.with_name("metadata.json")

    @property
    def last_language(self) -> Optional[str]:
        code = self.metadata.get("last_language") if self.metadata else None
        if isinstance(code, str) and code in SUPPORTED_LANGUAGES:
            return code
        return None


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
            if not path.is_dir():
                continue
            reference = path / "reference.wav"
            if reference.exists():
                metadata = self._read_metadata(path / "metadata.json")
                description = metadata.get("description", "") if metadata else ""
                yield VoiceProfile(path.name, reference, description, metadata=metadata)

    def _read_metadata(self, metadata_path: Path) -> Dict[str, object]:
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_metadata(
        self,
        voice: VoiceProfile,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        metadata = self._read_metadata(voice.metadata_path)
        metadata.update(
            {
                "speaker_id": voice.speaker_id,
                "description": voice.description,
            }
        )
        if extra:
            metadata.update(extra)
        voice.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def create_voice_profile(
        self,
        speaker_id: str,
        reference_wav: Path,
        description: str = "",
        language: Optional[str] = None,
    ) -> VoiceProfile:
        """Store a reference recording for a speaker."""

        destination = self._voice_dir(speaker_id)
        destination.mkdir(parents=True, exist_ok=True)
        stored_reference = destination / "reference.wav"
        shutil.copy2(reference_wav, stored_reference)
        profile = VoiceProfile(
            speaker_id=speaker_id,
            reference_path=stored_reference,
            description=description,
        )
        extra = {}
        if language:
            extra["last_language"] = normalise_language(language)
        self._write_metadata(profile, extra=extra or None)
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

        if not _HAS_SOUNDDEVICE:
            message = (
                "Microphone recording is unavailable because the 'sounddevice' package could not load "
                "the PortAudio library. Install the system PortAudio dependency (e.g. 'apt-get install "
                "libportaudio2') and reinstall sounddevice, or upload a reference wav file instead."
            )
            if _SOUNDDEVICE_IMPORT_ERROR is not None:
                raise MicrophoneUnavailableError(message) from _SOUNDDEVICE_IMPORT_ERROR
            raise MicrophoneUnavailableError(message)

        audio = sd.rec(int(self.sample_rate * self.record_seconds), samplerate=self.sample_rate, channels=1)
        sd.wait()
        sf.write(reference_file, audio, self.sample_rate)

        profile = VoiceProfile(
            speaker_id=speaker_id,
            reference_path=reference_file,
            description=description,
        )
        self._write_metadata(
            profile,
            extra={"last_language": canonical_language},
        )
        return profile

    def save_uploaded_reference(
        self,
        speaker_id: str,
        audio_path: Path,
        description: str = "",
        language: Optional[str] = None,
    ) -> VoiceProfile:
        """Persist a reference sample provided by the GUI."""

        return self.create_voice_profile(
            speaker_id,
            audio_path,
            description,
            language=language,
        )

    # ------------------------------------------------------------------
    # Synthesis helpers
    # ------------------------------------------------------------------
    def synthesize_to_file(
        self,
        speaker_id: str,
        text: str,
        language: str = "en",
        file_path: Optional[Path] = None,
        description: Optional[str] = None,
    ) -> Path:
        """Generate speech for the given text using the cloned voice."""

        canonical_language = normalise_language(language)
        tts_language_code = SUPPORTED_LANGUAGES[canonical_language]["tts_code"]

        voice_dir = self._voice_dir(speaker_id)
        reference = voice_dir / "reference.wav"
        if not reference.exists():
            raise FileNotFoundError(f"No reference audio found for speaker '{speaker_id}'.")

        if file_path is None:
            def _slugify(s: str) -> str:
                s = s.strip().lower()
                s = re.sub(r"\s+", "_", s)
                s = re.sub(r"[^a-z0-9_\-]", "", s)
                return s[:60] or "output"

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = (description or text[:40]).strip()
            file_name = f"{ts}_{_slugify(base)}.wav"
            file_path = voice_dir / file_name

        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(reference),
            language=tts_language_code,
            file_path=str(file_path),
        )
        return file_path

    def synthesize(self, speaker_id: str, text: str, language: str = "en", description: Optional[str] = None) -> tuple[int, np.ndarray]:
        """Generate speech and return it as raw audio."""

        file_path = self.synthesize_to_file(speaker_id, text, language, description=description)
        data, rate = sf.read(file_path, dtype="int16")
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
    "MicrophoneUnavailableError",
]
