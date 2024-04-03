from dataclasses import dataclass


@dataclass
class WhisperEncoderConfig:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int


@dataclass
class StableLMConfig:
    ...


@dataclass
class ProjectionConfig:
    ...