"""
Dataset classes for the silent-speech UML project.

EMGCharDataset         — reads preprocessed EMG tensors produced by
                         ``src/precompute_emg.py``.
LibriSpeechCharDataset — reads preprocessed audio tensors produced by
                         ``src/precompute_audio.py``.

Both datasets are thin cache readers: no signal processing, no FLAC decode,
no Wav2Vec2Processor, no absl FLAGS.  All preprocessing is done once by the
precompute scripts and the results are stored as fp16 tensors on disk.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_utils import TextTransform


# ---------------------------------------------------------------------------
# Vocab helpers shared by both datasets
# ---------------------------------------------------------------------------

def build_text_transform() -> TextTransform:
    """Return a TextTransform instance (same character set every time)."""
    return TextTransform()


def vocab_size(text_transform: TextTransform) -> int:
    """Number of output classes including the CTC blank token."""
    return len(text_transform.chars) + 1   # blank appended at index len(chars)


def blank_id(text_transform: TextTransform) -> int:
    return len(text_transform.chars)


# ---------------------------------------------------------------------------
# EMGCharDataset — reads tensors precomputed by src/precompute_emg.py
# ---------------------------------------------------------------------------

class EMGCharDataset(Dataset):
    """
    Per-sample dict:
        raw_emg          : FloatTensor (T_raw, 8)    — preprocessed 689 Hz EMG
        text_int         : LongTensor  (L,)          — char-level token ids
        lengths          : int — T_raw (CTC input length = T_raw // 8)
        text_int_lengths : int — L
    """

    def __init__(self, cache_path: str, split: str = 'train'):
        pt = Path(cache_path) / f'{split}.pt'
        if not pt.is_file():
            raise FileNotFoundError(
                f'EMG cache not found: {pt}\n'
                f'Run: python src/precompute_emg.py '
                f'--emg_data_dir <raw_dir> --out_dir {cache_path}'
            )
        payload = torch.load(pt, map_location='cpu')
        self.raw_emg_list:  list[torch.Tensor] = payload['raw_emg']   # fp16 (T_i, 8)
        self.text_int_list: list[torch.Tensor] = payload['text_int']  # int64 (L_i,)
        assert len(self.raw_emg_list) == len(self.text_int_list)

    def __len__(self) -> int:
        return len(self.raw_emg_list)

    def __getitem__(self, idx: int) -> dict:
        raw_emg  = self.raw_emg_list[idx]
        text_int = self.text_int_list[idx]
        return {
            'raw_emg':          raw_emg.float(),     # promote fp16 → fp32
            'text_int':         text_int.long(),
            'lengths':          raw_emg.shape[0],
            'text_int_lengths': text_int.shape[0],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        raw_emg_list = [b['raw_emg']  for b in batch]
        text_list    = [b['text_int'] for b in batch]
        lengths      = torch.tensor([b['lengths']          for b in batch])
        text_lengths = torch.tensor([b['text_int_lengths'] for b in batch])

        raw_emg_padded = pad_sequence(raw_emg_list, batch_first=True, padding_value=0.0)
        text_padded    = pad_sequence(text_list,    batch_first=True, padding_value=0)

        return {
            'raw_emg':          raw_emg_padded,   # (B, T_raw_max, 8)
            'text_int':         text_padded,      # (B, L_max)
            'lengths':          lengths,          # (B,) T_raw per sample
            'text_int_lengths': text_lengths,     # (B,)
        }


# ---------------------------------------------------------------------------
# LibriSpeechCharDataset — reads tensors precomputed by src/precompute_audio.py
# ---------------------------------------------------------------------------

class LibriSpeechCharDataset(Dataset):
    """
    Per-sample dict:
        audio_features   : FloatTensor (T_audio,)  — normalized waveform (fp32)
        text_int         : LongTensor  (L,)
        text_int_lengths : int
    """

    def __init__(self, cache_path: str, split: str = 'train-clean-100'):
        pt = Path(cache_path) / f'{split}.pt'
        if not pt.is_file():
            raise FileNotFoundError(
                f'Audio cache not found: {pt}\n'
                f'Run: python src/precompute_audio.py '
                f'--librispeech_dir <raw_dir> --out_dir {cache_path} '
                f'--splits {split}'
            )
        payload = torch.load(pt, map_location='cpu')
        self.audio_list:    list[torch.Tensor] = payload['audio']     # fp16 (T_i,)
        self.text_int_list: list[torch.Tensor] = payload['text_int']  # int64 (L_i,)
        assert len(self.audio_list) == len(self.text_int_list)

    def __len__(self) -> int:
        return len(self.audio_list)

    def __getitem__(self, idx: int) -> dict:
        audio    = self.audio_list[idx]
        text_int = self.text_int_list[idx]
        return {
            'audio_features':   audio.float(),       # promote fp16 → fp32
            'text_int':         text_int.long(),
            'text_int_lengths': text_int.shape[0],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        audio_list    = [b['audio_features'] for b in batch]
        text_list     = [b['text_int']       for b in batch]
        audio_lengths = torch.tensor([b['audio_features'].shape[0] for b in batch])
        text_lengths  = torch.tensor([b['text_int_lengths']        for b in batch])

        audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        text_padded  = pad_sequence(text_list,  batch_first=True, padding_value=0)

        return {
            'audio_features':   audio_padded,   # (B, T_audio_max)
            'audio_lengths':    audio_lengths,  # (B,)
            'text_int':         text_padded,    # (B, L_max)
            'text_int_lengths': text_lengths,   # (B,)
        }
