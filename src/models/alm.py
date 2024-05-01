from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.modeling_outputs


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
        """Performs tokens encoding by decoder's embedding layer

        :param tokens_batch: Input tokens.
        :return: 3d batched tensor of encoded tokens.
        """
        return self.decoder.model.embed_tokens(tokens_batch)
    
    def _encode_audio(self, mels_batch: torch.Tensor) -> torch.Tensor:
        """Performs mel-features encoding by sequent encoder and projection application.

        :param mels_batch: Batched mel-features.
        :return: 3d batched context matrix of input mel-features [B, C, T].
        """
        audio_context = self.encoder(mels_batch)
        if isinstance(audio_context, transformers.modeling_outputs.BaseModelOutput):
            audio_context = audio_context.last_hidden_state

        return self.proj(audio_context)

    def encode_multimodal_inputs(self, 
                                 mels: torch.Tensor, 
                                 tokens: torch.Tensor, 
                                 attention_mask: torch.Tensor) -> torch.Tensor:
        """Performs encoding of all the inputs with awareness to attention mask change.

        :param mels: Batched matrix of mel-filterbank features of size [B, T, C]
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
        # ...

        input_batch = torch.concat([enc_audio, enc_tokens], dim=1)

        # Extend attention mask according to audio encoding size
        T = enc_audio.shape[1]
        attention_mask = F.pad(attention_mask, [T, 0], mode="constant", value=1)

        return {
            "inputs_embeds": input_batch,
            "attention_mask": attention_mask,
        }
    
    def forward(self, 
                mels: torch.Tensor, 
                tokens: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs = self.encode_multimodal_inputs(mels, tokens, attention_mask)
        output = self.decoder(**inputs, return_dict=True)

        return output