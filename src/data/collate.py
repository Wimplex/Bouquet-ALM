from typing import Tuple, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence


class A2TCollator:
    def __init__(self, lm_ctx_size: int, lm_pad_token_id: int):
        self._ctx_size = lm_ctx_size
        self._pad_tok_id = lm_pad_token_id

    def __call__(self, batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs batchification on unaligned input.

        :param batch: 
        :return: Tuple of batched torch.Tensors for audio and textual features
        """
        mels, tokens = zip(*batch)
        mels_batch = torch.stack(mels, dim=0)
        tokens_batch = pad_sequence(tokens, padding_value=self._pad_tok_id).T

        return mels_batch, tokens_batch