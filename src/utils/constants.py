from pathlib import Path


# Weights paths
_WEIGHTS_PATH = Path("./weights")
WHISPER_BASE_ENCODER_PATH = _WEIGHTS_PATH / "whisper_base_encoder.pt"

# Samples paths
_SAMPLES_PATH = Path("./samples")
AUDIO_SAMPLE_PATH = _SAMPLES_PATH / "MiniLS-5694-64038-0003.flac"

# Assets path
_ASSETS_PATH = Path("./assets")
MFB_PATH = _ASSETS_PATH / "mel_filters.npz"