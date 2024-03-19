import torch

from src.models.whisper import WhisperEncoder, ModelDimensions
from src.utils.audio import log_mel_spectrogram
from src.utils.basic import filter_out_state_dict_keys
from src.utils.constants import WHISPER_BASE_ENCODER_PATH, AUDIO_SAMPLE_PATH


# Load ckpt
checkpoint = torch.load(WHISPER_BASE_ENCODER_PATH, "cpu")

# Weights filtration machinery
# checkpoint["model_state_dict"] = filter_out_state_dict_keys(checkpoint["model_state_dict"], "encoder")
# torch.save(checkpoint, "./weights/whisper_base_encoder.pt")
# exit()

# Load model
dims = ModelDimensions(**checkpoint["dims"])
encoder = WhisperEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
encoder.load_state_dict(checkpoint["model_state_dict"])
# print(encoder)

# Load audio
mel_tensor = log_mel_spectrogram(str(AUDIO_SAMPLE_PATH), dims.n_mels)
print(mel_tensor.shape)

encoder(mel_tensor.unsqueeze(0))