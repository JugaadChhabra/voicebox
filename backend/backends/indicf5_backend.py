"""
IndicF5 TTS backend implementation.

Wraps the AI4Bharat IndicF5 model for high-quality Indic language
text-to-speech with voice cloning support.

IndicF5 is based on F5-TTS (0.4B params, flow matching) and supports
11 Indian languages:
  Assamese (as), Bengali (bn), Gujarati (gu), Hindi (hi),
  Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or),
  Punjabi (pa), Tamil (ta), Telugu (te)

Voice cloning works via reference audio (~10-15s sample + transcript).
Output is 24kHz mono WAV.

Requires: pip install git+https://github.com/ai4bharat/IndicF5.git

Model weights (~1.4GB) are downloaded from HuggingFace on first use.
The Vocos vocoder (~50MB) is also downloaded separately on first inference.

The model is gated — users must accept terms at:
  https://huggingface.co/ai4bharat/IndicF5
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    get_torch_device,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt

logger = logging.getLogger(__name__)

INDICF5_HF_REPO = "ai4bharat/IndicF5"
INDICF5_SAMPLE_RATE = 24000

# Supported Indic languages: ISO 639-1 code -> language name
INDICF5_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}


def _check_f5tts_available() -> bool:
    """Check if the f5_tts package is installed."""
    try:
        import f5_tts  # noqa: F401

        return True
    except ImportError:
        return False


class IndicF5TTSBackend:
    """IndicF5 TTS backend — Indic language voice cloning via F5-TTS.

    Uses the ``f5_tts.api.F5TTS`` class for full control over inference
    (seed, device, speed). Falls back to ``transformers.AutoModel`` if
    the f5_tts package is not installed.
    """

    def __init__(self):
        self._model = None
        self._use_f5_api: bool = _check_f5tts_available()
        self._device: Optional[str] = None
        self.model_size = "default"

    def _get_device(self) -> str:
        """Select device. IndicF5 supports CUDA, MPS (Apple Silicon), and CPU."""
        return get_torch_device(allow_mps=True)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return INDICF5_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check if IndicF5 model files are cached locally."""
        from .base import is_model_cached

        return is_model_cached(INDICF5_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        """Load the IndicF5 model."""
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "indicf5"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            device = self.device
            logger.info("Loading IndicF5 on %s...", device)

            if self._use_f5_api:
                self._load_via_f5_api(device)
            else:
                self._load_via_automodel()

        logger.info("IndicF5 loaded successfully")

    def _load_via_f5_api(self, device: str):
        """Load using the f5_tts.api.F5TTS class (preferred)."""
        from f5_tts.api import F5TTS

        self._model = F5TTS(
            model_type="F5-TTS",
            ckpt_file="",  # auto-downloads from HuggingFace
            vocab_file="",  # auto-downloads from HuggingFace
            ode_method="euler",
            use_ema=True,
            vocoder_name="vocos",
            device=device,
        )
        logger.info("IndicF5 loaded via F5TTS API on %s", device)

    def _load_via_automodel(self):
        """Fallback: load using transformers AutoModel."""
        from transformers import AutoModel

        logger.warning(
            "f5_tts package not found — falling back to AutoModel. "
            "Install for better integration: "
            "pip install git+https://github.com/ai4bharat/IndicF5.git"
        )
        self._model = AutoModel.from_pretrained(
            INDICF5_HF_REPO,
            trust_remote_code=True,
        )

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have empty_cache, but deleting the model frees memory
                pass

            logger.info("IndicF5 unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        IndicF5 uses reference audio + transcript for voice cloning.
        Reference audio should be 10-15 seconds (auto-clipped to 15s).

        IMPORTANT: reference_text must not be empty — an empty string
        triggers IndicF5 to download whisper-large-v3-turbo (~3GB)
        for auto-transcription.
        """
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None and isinstance(cached_prompt, dict):
                cached_audio = cached_prompt.get("ref_audio_path")
                if cached_audio and Path(cached_audio).exists():
                    return cached_prompt, True

        voice_prompt = {
            "ref_audio_path": str(audio_path),
            "ref_text": reference_text,
        }

        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt)

        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> Tuple[np.ndarray, str]:
        """Combine voice prompts — uses base implementation for audio concatenation."""
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=INDICF5_SAMPLE_RATE
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "hi",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using IndicF5.

        Args:
            text: Text to synthesize (in any supported Indic language)
            voice_prompt: Dict with ref_audio_path and ref_text keys
            language: Language code (as, bn, gu, hi, kn, ml, mr, or, pa, ta, te)
            seed: Random seed for reproducibility
            instruct: Not supported by IndicF5 (ignored)

        Returns:
            Tuple of (audio_array as float32 numpy, sample_rate)
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio_path") or voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text", "")

        if ref_audio and not Path(ref_audio).exists():
            logger.warning("Reference audio not found: %s", ref_audio)
            raise ValueError(
                f"Reference audio file not found: {ref_audio}. "
                "IndicF5 requires reference audio for voice cloning."
            )

        if not ref_audio:
            raise ValueError(
                "IndicF5 requires reference audio for generation. "
                "Please upload a voice sample to the profile."
            )

        # Guard against empty ref_text triggering a whisper download
        if not ref_text or not ref_text.strip():
            ref_text = "."
            logger.warning(
                "Empty reference text — using placeholder to avoid "
                "auto-transcription download."
            )

        if self._use_f5_api:
            return await asyncio.to_thread(
                self._generate_f5_api, text, ref_audio, ref_text, seed
            )
        else:
            return await asyncio.to_thread(
                self._generate_automodel, text, ref_audio, ref_text, seed
            )

    def _generate_f5_api(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        """Generate using the F5TTS API (preferred path)."""
        logger.info("Generating Indic audio (F5TTS API) for: %s", text[:80])

        wav, sr, _spect = self._model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            seed=seed if seed is not None else -1,  # -1 = random
            nfe_step=32,
            speed=1.0,
            cross_fade_duration=0.15,
        )

        audio = np.asarray(wav, dtype=np.float32).squeeze()

        if audio.size == 0:
            logger.warning("IndicF5 returned empty audio, returning silence")
            return np.zeros(INDICF5_SAMPLE_RATE, dtype=np.float32), INDICF5_SAMPLE_RATE

        return audio, sr

    def _generate_automodel(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        """Generate using the AutoModel fallback."""
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        logger.info("Generating Indic audio (AutoModel) for: %s", text[:80])

        audio = self._model(text, ref_audio_path=ref_audio, ref_text=ref_text)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio).squeeze()

        # AutoModel may return int16 — normalize to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        if audio.size == 0:
            logger.warning("IndicF5 returned empty audio, returning silence")
            return np.zeros(INDICF5_SAMPLE_RATE, dtype=np.float32), INDICF5_SAMPLE_RATE

        return audio, INDICF5_SAMPLE_RATE
