from pathlib import Path


# Samples paths
_SAMPLES_PATH = Path("./samples")
AUDIO_SAMPLE_PATH = _SAMPLES_PATH / "MiniLS-5694-64038-0003.flac"

# Assets path
__ASSETS_PATH = Path("../../../assets")
_ASSETS_PATH = Path("./assets")
MFB_PATH = _ASSETS_PATH / "mel_filters.npz"
# STABLELM_PATH = _ASSETS_PATH / "stablelm_2_zephyr_1_6.stablelm"
# STABLELM_TOKENIZER_PATH = _ASSETS_PATH / "stablelm_2_tokenizer.json"
# WHISPER_BASE_ENCODER_PATH = _ASSETS_PATH / "whisper_base_encoder.pt"

# Tokens
AUDIO_TOKENS = {
    "boa_token": "<|startofaudio|>", 
    "eoa_token": "<|endofaudio|>"
}