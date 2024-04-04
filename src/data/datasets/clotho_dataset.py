from typing import Union, Iterable
from pathlib import Path

import pandas as pd
import torch

from .base_dataset import BaseA2TDataset
from src.utils.basic import cast


def read_clotho_manifest(manifest_path: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, index_col="file_name")
    df = df[df.index.str.endswith(".wav")]

    return df


class ClothoDataset(BaseA2TDataset):
    def __init__(self, 
                 manifest_path: Union[Path, str], 
                 audio_path: Union[Path, str], **kwargs):
        self._df = read_clotho_manifest(manifest_path)
        self._audio_path = cast(audio_path, Path)
        super().__init__(**kwargs)

    def build_key2text(self) -> torch.Dict[str, Iterable[str]]:
        key2text = {}
        for key, caps in self._df.iterrows():
            key2text[key] = caps.values.tolist()

        return key2text

    def build_key2audio(self) -> torch.Dict[str, str]:
        return {key: self._audio_path / key for key in self._df.index.tolist()}