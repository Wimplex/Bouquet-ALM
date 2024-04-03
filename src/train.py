import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.alm import ALM, ALM_SETTINGS
from src.utils.config import *
from src.utils.constants import WHISPER_BASE_ENCODER_PATH


# class ALM_Dataset(Dataset):
#     def __init__(self):
#         ...

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         ...

#     def __len__(self) -> int:
#         ...


def main():

    model_type = "camellia"
    model = ALM(ALM_SETTINGS[model_type])
    
    # Forward model
    bsize = 2
    mels = torch.randn([bsize, 80, 3000])
    texts = "First sentence, here we go!<|batch_sep|>A-a-and the second one!"
    out = model(mels, texts)
    print(out.__dict__.keys())

if __name__ == "__main__":
    main()