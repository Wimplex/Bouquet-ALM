import torch
import torch.nn.functional as F

from src.models.whisper import WhisperEncoder, WhisperEncoderConfig
from src.utils.audio import log_mel_spectrogram
from src.utils.basic import filter_out_state_dict_keys
from src.utils.constants import WHISPER_BASE_ENCODER_PATH, AUDIO_SAMPLE_PATH


# Load ckpt
checkpoint = torch.load(WHISPER_BASE_ENCODER_PATH, "cpu")

# Weights filtration machinery
# checkpoint["model_state_dict"] = filter_out_state_dict_keys(checkpoint["model_state_dict"], "encoder")
# del checkpoint["dims"]["n_text_ctx"]
# del checkpoint["dims"]["n_text_state"]
# del checkpoint["dims"]["n_text_head"]
# del checkpoint["dims"]["n_text_layer"]
# del checkpoint["dims"]["n_vocab"]
# torch.save(checkpoint, "./assets/whisper_base_encoder.pt")
# exit()

# Load model
dims = WhisperEncoderConfig(**checkpoint["dims"])
encoder = WhisperEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
encoder.load_state_dict(checkpoint["model_state_dict"])
# print(encoder)

# Load audio
mel_tensor = log_mel_spectrogram(str(AUDIO_SAMPLE_PATH), dims.n_mels)
pad_size = 3000 - mel_tensor.shape[-1]
mel_tensor = F.pad(mel_tensor, [0, pad_size], value=0)
print(mel_tensor.shape)

print(encoder(mel_tensor.unsqueeze(0)).shape)