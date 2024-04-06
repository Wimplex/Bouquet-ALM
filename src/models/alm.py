from typing import Iterable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from src.models.whisper import WhisperEncoder


SPECIAL_AUDIO_TOKENS = {
    "boa_token": "<|startofaudio|>", 
    "eoa_token": "<|endofaudio|>"
}


class ALM(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 projection: nn.Module, 
                 decoder: transformers.PreTrainedModel):
        super().__init__()

        self.encoder: nn.Module = encoder
        self.proj: nn.Module = projection
        self.decoder: transformers.PreTrainedModel = decoder

    def _encode_tokens(self, tokens_batch: torch.Tensor) -> torch.Tensor:
        """Performs tokens encoding by decoder's Embedding layer

        :param tokens_batch: Input tokens.
        :return: 3d batched tensor of encoded tokens.
        """
        return self.decoder.model.embed_tokens(tokens_batch)
    
    def _encode_audio(self, mels_batch: torch.Tensor) -> torch.Tensor:
        """Performs mel-features encoding by sequent encoder and projection application.

        :param mels_batch: Batched mel-features.
        :return: 3d batched context matrix of input mel-features.
        """
        return self.proj(self.encoder(mels_batch))

    def encode_multimodal_inputs(self, 
                                 mels: torch.Tensor, 
                                 tokens: torch.Tensor, 
                                 attention_mask: torch.Tensor) -> torch.Tensor:
        """Performs encoding of all the inputs with awareness to attention mask change.

        :param mels: _description_
        :type mels: torch.Tensor
        :param tokens: _description_
        :type tokens: torch.Tensor
        :param attention_mask: _description_
        :type attention_mask: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """
        enc_audio = self._encode_audio(mels)
        enc_tokens = self._encode_tokens(tokens)

        # TODO: отделить аудио-часть специальным токеном (или их последовательностью)
        ...

        input_batch = torch.concat([enc_audio, enc_tokens], dim=1)

        # Extend attention mask according to audio encoding size
        T = enc_audio.shape[-1]
        attention_mask = F.pad(attention_mask, [T, 0], mode="constant", value=1)

        return {
            "input_embeds": input_batch,
            "attention_mask": attention_mask,
        }
    
    def forward(self, 
                mels: torch.Tensor, 
                tokens: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs = self.encode_multimodal_inputs(mels, tokens, attention_mask)
        output = self.decoder(**inputs)

        return output


# class _ALM(nn.Module):

#     encoder: nn.Module
#     decoder: AutoModelForCausalLM
#     projection: nn.Module
#     tokenizer: AutoTokenizer

#     def __init__(self, config: ALMConfig):
#         super().__init__()

#         self._init_encoder(config)
#         self._init_projection(config)
#         self._init_tokenizer(config)
#         self._init_decoder(config)

#         self._extend_vocab()

#     def _init_encoder(self, config: ALMConfig):
#         self.encoder = WhisperEncoder(
#             config.whisper_n_mels, 
#             config.whisper_n_audio_ctx, 
#             config.whisper_n_audio_state, 
#             config.whisper_n_audio_head, 
#             config.whisper_n_audio_layer,
#         )

#         ckpt = torch.load(config.whisper_ckpt_path, map_location="cpu")
#         self.encoder.load_state_dict(ckpt["model_state_dict"])
    
#     def _init_projection(self, config: ALMConfig):
#         # TODO: Fix that
#         self.projection = nn.Linear(512, 2048)

#     def _init_tokenizer(self, config: ALMConfig):
#         self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
 
#     def _init_decoder(self, config: ALMConfig):
#         self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_model_name, device_map="cpu")

#     def _extend_vocab(self):
#         for new_token in SPECIAL_AUDIO_TOKENS:
#             if new_token not in self.tokenizer.vocab:
#                 self.tokenizer.add_tokens(new_token)
        
#         if self.decoder.model.embed_tokens.weight.shape[0] != len(self.tokenizer):
#             self.decoder.resize_token_embeddings(len(self.tokenizer))

#     def _get_special_audio_tokens_embeddings(self,
#                                              batch_size: int,
#                                              device: str) -> Dict[str, torch.Tensor]:
#         emb_dim_size = self.decoder.model.embed_tokens.weight.shape[-1]
#         token2emb = {}
#         for str_token in ["boa_token", "eoa_token"]:
#             token_id = torch.tensor([self.tokenizer.vocab[str_token]], device=device)
#             token_emb = self.decoder.model.embed_tokens(token_id).expand(batch_size, emb_dim_size).unsqueeze(1)
#             token2emb[str_token] = token_emb

#         return token2emb
    
#     def _encode_prompts(self, prompts: Iterable[str], device: str) -> torch.Tensor:
#         texts = prompts.split("<|batch_sep|>")
#         tokens = []
#         for text in texts:
#             tokens.append(self.tokenizer(text)["input_ids"])

#         max_len = max(map(len, tokens))
#         attention_mask = []
#         for i in range(len(tokens)):
#             pad_size = max_len - len(tokens[i]) + 1
#             attention_mask.append(len(tokens[i]) * [1] + pad_size * [0])
#             tokens[i] = tokens[i] + pad_size * [self.tokenizer.eos_token_id]

#         tokens = torch.tensor(tokens, dtype=torch.int32, device=device)
#         tokens_emb = self.decoder.model.embed_tokens(tokens)

#         return tokens_emb, torch.tensor(attention_mask, dtype=torch.int32, device=device)
    
#     def _encode_mels(self, mels: torch.Tensor) -> torch.Tensor:
#         return self.projection(self.encoder(mels))

#     def encode_mm(self, mels: torch.Tensor, prompts: Iterable[str]) -> Dict[str, torch.Tensor]:
#         # Encode prompts
#         text_part, attention_mask = self._encode_prompts(prompts, mels.device)
        
#         # Encode audio
#         audio_embs = self._encode_mels(mels)
#         special_audio_embs = self._get_special_audio_tokens_embeddings(mels.shape[0], mels.device)
#         audio_part = torch.concat([
#             special_audio_embs["boa_token"], 
#             audio_embs, 
#             special_audio_embs["eoa_token"]
#         ], dim=1)

#         # Update attention mask
#         attention_mask = F.pad(attention_mask, [audio_part.shape[1], 0], value=1)

#         return {
#             "inputs_embeds": torch.concatenate([audio_part, text_part], dim=1), 
#             "attention_mask": attention_mask
#         }

#     def forward(self, mels: torch.Tensor, prompts: Iterable[str]):
#         mm_input = self.encode_mm(mels, prompts)
#         out = self.decoder(**mm_input)

#         return out