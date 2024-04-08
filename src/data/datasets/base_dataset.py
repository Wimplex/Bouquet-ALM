import random
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union, Iterable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils.audio import log_mel_spectrogram


class BaseA2TDataset(Dataset, ABC):
    def __init__(self,
                 *,
                 tokenizer: PreTrainedTokenizer,
                 audio_ctx_size: int, 
                 audio_n_mels: int):        
        self.tokenizer = tokenizer
        self.audio_ctx_size = audio_ctx_size
        self.audio_n_mels = audio_n_mels

        self.key2text = self.build_key2text()
        self.key2audio = self.build_key2audio()
        self._keys = list(self.key2text.keys())

    @abstractmethod
    def build_key2text(self) -> Dict[str, Union[str, Iterable[str]]]:
        raise NotImplementedError()
    
    @abstractmethod
    def build_key2audio(self) -> Dict[str, str]:
        raise NotImplementedError()
    
    def encode_text(self, key: str) -> str:
        """Performs text preprocessing"""
        text = self.key2text[key]
        if isinstance(text, Iterable):
            text = random.choice(text)
    
        return torch.tensor(
            self.tokenizer(text)["input_ids"],
            dtype=torch.long
        )
        
    def encode_audio(self, key: str) -> str:
        """Performs audio preprocessing"""
        mels = log_mel_spectrogram(str(self.key2audio[key]), n_mels=self.audio_n_mels)
        mels = torch.nn.functional.pad(
            mels, [0, self.audio_ctx_size - mels.shape[-1]], mode="constant", value=0)

        return mels

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._keys[idx]
        tokens = self.encode_text(key)
        mels = self.encode_audio(key)

        return mels, tokens