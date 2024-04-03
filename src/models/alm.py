from typing import Iterable, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.whisper import WhisperEncoder
from src.utils.constants import WHISPER_BASE_ENCODER_PATH


@dataclass
class ALMConfig:
    # Whisper related
    whisper_n_mels: int
    whisper_n_audio_ctx: int
    whisper_n_audio_state: int
    whisper_n_audio_head: int
    whisper_n_audio_layer: int
    whisper_ckpt_path: str

    # LLM decoder related
    decoder_model_name: str

    # Projection network related
    # ...


ALM_SETTINGS = {
    # Petite
    "bluebell": ...,

    # Tiny
    "camellia": ALMConfig(
        whisper_n_mels=80,
        whisper_n_audio_ctx=1500,
        whisper_n_audio_state=512,
        whisper_n_audio_head=8,
        whisper_n_audio_layer=6,
        whisper_ckpt_path=WHISPER_BASE_ENCODER_PATH,
        decoder_model_name="stabilityai/stablelm-2-zephyr-1_6b",
    ),

    # Small
    "lilac": ...,
}


SPECIAL_AUDIO_TOKENS = {
    "boa_token": "<|startofaudio|>", 
    "eoa_token": "<|endofaudio|>"
}


class ALM(nn.Module):

    encoder: nn.Module
    decoder: AutoModelForCausalLM
    projection: nn.Module
    tokenizer: AutoTokenizer

    def __init__(self, config: ALMConfig):
        super().__init__()

        self._init_encoder(config)
        self._init_projection(config)
        self._init_tokenizer(config)
        self._init_decoder(config)

        self._extend_vocab()

    def _init_encoder(self, config: ALMConfig):
        self.encoder = WhisperEncoder(
            config.whisper_n_mels, 
            config.whisper_n_audio_ctx, 
            config.whisper_n_audio_state, 
            config.whisper_n_audio_head, 
            config.whisper_n_audio_layer,
        )

        ckpt = torch.load(config.whisper_ckpt_path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["model_state_dict"])
    
    def _init_projection(self, config: ALMConfig):
        # TODO: Fix that
        self.projection = nn.Linear(512, 2048)

    def _init_tokenizer(self, config: ALMConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
 
    def _init_decoder(self, config: ALMConfig):
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_model_name, device_map="cpu")

    def _extend_vocab(self):
        for new_token in SPECIAL_AUDIO_TOKENS:
            if new_token not in self.tokenizer.vocab:
                self.tokenizer.add_tokens(new_token)
        
        if self.decoder.model.embed_tokens.weight.shape[0] != len(self.tokenizer):
            self.decoder.resize_token_embeddings(len(self.tokenizer))

    def _get_special_audio_tokens_embeddings(self,
                                             batch_size: int,
                                             device: str) -> Dict[str, torch.Tensor]:
        emb_dim_size = self.decoder.model.embed_tokens.weight.shape[-1]
        token2emb = {}
        for str_token in ["boa_token", "eoa_token"]:
            token_id = torch.tensor([self.tokenizer.vocab[str_token]], device=device)
            token_emb = self.decoder.model.embed_tokens(token_id).expand(batch_size, emb_dim_size).unsqueeze(1)
            token2emb[str_token] = token_emb

        return token2emb
    
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

        return tokens_emb, torch.tensor(attention_mask, dtype=torch.int32, device=device)
    
    def _encode_mels(self, mels: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(mels))

    def encode_mm(self, mels: torch.Tensor, prompts: Iterable[str]) -> Dict[str, torch.Tensor]:
        # Encode prompts
        text_part, attention_mask = self._encode_prompts(prompts, mels.device)
        
        # Encode audio
        audio_embs = self._encode_mels(mels)
        special_audio_embs = self._get_special_audio_tokens_embeddings(mels.shape[0], mels.device)
        audio_part = torch.concat([
            special_audio_embs["boa_token"], 
            audio_embs, 
            special_audio_embs["eoa_token"]
        ], dim=1)

        # Update attention mask
        attention_mask = F.pad(attention_mask, [audio_part.shape[1], 0], value=1)

        return {
            "inputs_embeds": torch.concatenate([audio_part, text_part], dim=1), 
            "attention_mask": attention_mask
        }

    def forward(self, mels: torch.Tensor, prompts: Iterable[str]):
        mm_input = self.encode_mm(mels, prompts)
        out = self.decoder(**mm_input)

        return out