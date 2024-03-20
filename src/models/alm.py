from typing import Iterable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


SPECIAL_AUDIO_TOKENS = {
    "boa_token": "<|startofaudio|>", 
    "eoa_token": "<|endofaudio|>"
}


class ALM(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        projection: nn.Module,
        decoder: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer
    ):
        super().__init__()

        self.encoder: nn.Module = encoder
        self.projection: nn.Module = projection
        self.decoder: AutoModelForCausalLM = decoder
        self.tokenizer: AutoTokenizer = tokenizer

        self._extend_vocab()
 
    def _extend_vocab(self):
        for new_token in SPECIAL_AUDIO_TOKENS:
            if new_token not in self.tokenizer.vocab:
                self.tokenizer.add_tokens(new_token)
        
        if self.decoder.model.embed_tokens.weights.shape[0] != len(self.tokenizer):
            self.decoder.resize_token_embeddings(len(self.tokenizer))

    def _get_special_audio_tokens_embeddings(
        self, 
        batch_size: int,
        device: str
    ) -> Dict[str, torch.Tensor]:
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