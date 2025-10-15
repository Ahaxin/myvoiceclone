"""Voice cloning package."""

from .clone import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    LANGUAGE_ALIASES,
    REFERENCE_PROMPTS,
    SUPPORTED_LANGUAGES,
    VoiceCloneService,
    get_reference_prompt,
    normalise_language,
)

__all__ = [
    "VoiceCloneService",
    "SUPPORTED_LANGUAGES",
    "AVAILABLE_ENGINES",
    "DEFAULT_ENGINE",
    "LANGUAGE_ALIASES",
    "normalise_language",
    "get_reference_prompt",
    "REFERENCE_PROMPTS",
]
