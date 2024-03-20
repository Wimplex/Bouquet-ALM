import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.constants import WHISPER_BASE_ENCODER_PATH
from src.models.whisper import WhisperEncoder, WhisperEncoderConfig





class ALM_Dataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ...

    def __len__(self) -> int:
        ...


def main():
    # Load whisper encoder
    checkpoint = torch.load(WHISPER_BASE_ENCODER_PATH, "cpu")
    dims = WhisperEncoderConfig(**checkpoint["dims"])
    encoder = WhisperEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    
    # Load stablelm
    model_name = "stabilityai/stablelm-2-zephyr-1_6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    # Init model
    model = ALM(encoder, nn.Linear(512, 2048), lm, tokenizer)

    # Forward model
    bsize = 2
    mels = torch.randn([bsize, 80, 3000])
    texts = "Hello!<|batch_sep|>Hi, how are you?"
    out = model(mels, texts)
    print(out.shape)

if __name__ == "__main__":
    main()