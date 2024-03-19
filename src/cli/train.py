from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors import safe_open
from torch.utils.data import Dataset
from transformers import AutoTokenizer, StableLmForCausalLM, AutoModelForCausalLM

from src.utils.audio import log_mel_spectrogram
from src.utils.constants import WHISPER_BASE_ENCODER_PATH
from src.models.whisper import WhisperEncoder, WhisperEncoderConfig


class ALM(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        projection: nn.Module, 
        decoder: StableLmForCausalLM, 
        tokenizer: AutoTokenizer
    ):
        super().__init__()

        self.encoder = encoder
        self.projection = projection
        self.decoder: StableLmForCausalLM = decoder
        self.tokenizer: AutoTokenizer = tokenizer

        self._special_audio_tokens = {"boa_token": "<|startofaudio|>", "eoa_token": "<|endofaudio|>"}

        self._extend_vocab()

    @property
    def decoder_ctx_size(self) -> int:
        return self.decoder.model.embed_tokens.weight.shape[-1]
 
    def _extend_vocab(self):
        for new_token in self._special_audio_tokens:
            if new_token not in self.tokenizer.vocab:
                self.tokenizer.add_tokens(new_token)
        
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def _get_special_audio_tokens_embeddings(self, device: str) -> torch.Tensor:
        # boa_idx = torch.tensor([self.tokenizer.vocab[]], device=device)
        boa_tok = torch.tensor([self.tokenizer.vocab["boa_token"]], device=device)
        eoa_tok = torch.tensor([self.tokenizer.vocab["eoa_token"]], device=device)

        return self.decoder.model.embed_tokens(torch.stack([boa_tok, eoa_tok]))
    
    def _encode_prompts(self, prompts: Iterable[str], device: str) -> torch.Tensor:
        texts = prompts.split("<|batch_sep|>")
        tokens = []
        for text in texts:
            tokens.append(self.tokenizer(text)["input_ids"])

        max_len = max(map(len, tokens))
        attention_mask = []
        for i in range(len(tokens)):
            pad_size = max_len - len(tokens[i]) + 1
            attention_mask.append(len(tokens[i]) * [1] + pad_size * [0])
            tokens[i] = tokens[i] + pad_size * [self.tokenizer.eos_token_id]

        tokens = torch.tensor(tokens, dtype=torch.int32, device=device)
        tokens_emb = self.decoder.model.embed_tokens(tokens)

        return tokens_emb, attention_mask

    def encode_mm(self, mels: torch.Tensor, prompts: Iterable[str]):
        audio_embs = self.projection(self.encoder(mels))
        text_embs = self._encode_prompts(prompts, mels.device)

        special_audio_embs = self._get_special_audio_tokens_embeddings(mels.device)

        mm_input = torch.concatenate([
            special_audio_embs[0].unsqueeze(0), 
            audio_embs, 
            special_audio_embs[1].unsqueeze(0), 
            text_embs
        ])

        return mm_input

    def forward(self, mels: torch.Tensor, prompts: Iterable[str]):
        mm_input = self.encode_mm(mels, prompts)

        # Тут на mm_input нужно добавить позиционное кодирование.
        # Похоже придётся тащить код stablelm
        # self.decoder(mm_input)

        return mm_input


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
    model(mels, texts)

if __name__ == "__main__":
    main()