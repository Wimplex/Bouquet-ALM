import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (PreTrainedModel, 
                          PreTrainedTokenizer,
                          AutoModel,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          WhisperConfig, 
                          LlamaConfig)
from einops import rearrange

from src.data.datasets import ClothoDataset
from src.data.collate import A2TCollator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__file__)


def init_whisper_encoder(size: str = "tiny") -> PreTrainedModel:
    model_name = f"openai/whisper-{size}"
    cfg = WhisperConfig.from_pretrained(model_name)
    cfg.decoder_layers = 0

    return AutoModel.from_pretrained(model_name, config=cfg).encoder


def init_tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def init_llm_decoder(model_name: str) -> PreTrainedModel:

    # ###
    cfg = LlamaConfig.from_pretrained(model_name)
    cfg.num_hidden_layers = 2
    # ###

    return AutoModelForCausalLM.from_pretrained(model_name, config=cfg)


def freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def train(encoder: nn.Module,
          projection: nn.Module,
          decoder: nn.Module,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          train_loader: DataLoader,
          num_epochs: int = 10,
          log_period: int = 10,
          device: str = "cuda:0") -> nn.Module:
    
    # Prepare encoder/decoder for training
    encoder.to(device)
    encoder.train()
    freeze_model(encoder)

    decoder.to(device)
    decoder.train()
    freeze_model(decoder)

    # Leave projection matrix unfrozen
    projection.to(device)
    projection.train()


    for ep in range(num_epochs):
        logger.info(f"#{ep} epoch started.")

        for i, item in enumerate(train_loader):
            mels, tokens, attn_mask = item
            mels = mels.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            # Encode audio
            projected_audio_embs = projection(encoder(mels).last_hidden_state)

            # Prepare decoder inputs
            input_tokens = tokens[..., :-1]
            target_tokens = tokens[..., 1:]
            attn_mask = attn_mask[..., :-1]
            encoded_input_tokens = decoder.model.embed_tokens(input_tokens)
            input_batch = torch.concat([projected_audio_embs, encoded_input_tokens], dim=1)

            # Expand attention matrix to the left
            T = projected_audio_embs.shape[1]
            attn_mask = F.pad(attn_mask, [T, 0], mode="constant", value=1)

            # Predict tokens
            model_inputs = {
                "inputs_embeds": input_batch,
                "attention_mask": attn_mask
            }
            output = decoder(**model_inputs)

            # Compute loss only for textual part
            logits = output["logits"][:, -input_tokens.shape[-1]:]
            loss = criterion(
                rearrange(logits, "b t c -> (b t) c"), 
                rearrange(target_tokens, "b t -> (b t)")
            )
            loss.backward()
            optimizer.step()
    
            if i != 0 and i % log_period == 0:
                msg = f"train_loss: {round(loss.detach().item(), 3)}"
                logger.info(msg)

    return encoder, projection, decoder


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    datasets_root_dir = Path("../../Datasets/speech")

    # Init tokenizer
    tokenizer = init_tokenizer(model_name)

    # Init dataloader
    manifest_path = datasets_root_dir / "Captioning/clotho_audio_development/clotho_captions_development.csv"
    audio_path = datasets_root_dir / "Captioning/clotho_audio_development/development"
    dataset = ClothoDataset(
        manifest_path, 
        audio_path, 
        audio_ctx_size=3000, 
        audio_n_mels=80, 
        tokenizer=tokenizer
    )
    train_loader = DataLoader(dataset, 4, shuffle=True, collate_fn=A2TCollator(2048, 0))

    # Init whisper and projection
    encoder = init_whisper_encoder("tiny")
    logger.info("Whisper initialized")

    # Init projection
    proj = nn.Linear(384, 2048)
    logger.info("Projection matrix initialized")

    # Init decoder
    decoder = init_llm_decoder(model_name)
    logger.info("Decoder initialized")

    # Init criterion
    criterion = nn.CrossEntropyLoss()

    # Init optimizer
    optimizer = optim.AdamW(proj.parameters(), lr=5e-5)

    # Run training
    train(encoder, proj, decoder, criterion, optimizer, train_loader, 10, 10, device="cuda:0")


if __name__ == "__main__":
    main()