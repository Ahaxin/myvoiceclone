"""Voice cloning logic built on top of Coqui TTS."""

from __future__ import annotations

import json
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
        record_seconds: int = 10,
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
    def record_reference(self, speaker_id: str, description: str = "") -> VoiceProfile:
        """Record a reference sample from the microphone."""

        directory = self._voice_dir(speaker_id)
        directory.mkdir(parents=True, exist_ok=True)
        reference_file = directory / "reference.wav"

        print(
            "Recording reference for speaker '%s'. Please read a short paragraph clearly.\n"
            "Recording will last %s seconds..." % (speaker_id, self.record_seconds)
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

        canonical_language = self._normalise_language(language)
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

    def _normalise_language(self, language: str) -> str:
        code = language.lower()
        if code in SUPPORTED_LANGUAGES:
            return code
        alias = LANGUAGE_ALIASES.get(code)
        if alias and alias in SUPPORTED_LANGUAGES:
            return alias
        raise ValueError(
            f"Unsupported language '{language}'. Choose from: {', '.join(SUPPORTED_LANGUAGES)}"
        )


__all__ = [
    "VoiceCloneService",
    "SUPPORTED_LANGUAGES",
    "LANGUAGE_ALIASES",
    "AVAILABLE_ENGINES",
    "DEFAULT_ENGINE",
    "VoiceProfile",
]
