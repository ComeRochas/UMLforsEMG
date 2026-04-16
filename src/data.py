"""
Dataset classes for the silent-speech UML project.

EMGCharDataset         — reads preprocessed EMG tensors produced by
                         ``src/precompute_emg.py``.  No signal processing, no
                         absl FLAGS, no audio — the bare minimum needed for
                         EMG → CTC training.
LibriSpeechCharDataset — loads LibriSpeech + normalizes waveforms via
                         Wav2Vec2Processor (feature extractor only).
                         Both datasets share the same TextTransform / vocab.
"""
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor

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
# LibriSpeechCharDataset
# ---------------------------------------------------------------------------

class LibriSpeechCharDataset(Dataset):
    """
    Loads LibriSpeech audio + transcriptions.

    The raw waveform is normalized by Wav2Vec2Processor (feature extractor
    only — no model forward pass here).  The wav2vec2 encoder is applied
    inside AudioEncoder at training time.

    Per-sample dict:
        audio_features   : FloatTensor (T_audio,)  — normalized waveform
        text_int         : LongTensor  (L,)
        text_int_lengths : int
    """

    WAV2VEC2_MODEL = 'facebook/wav2vec2-base'
    TARGET_SR      = 16_000

    def __init__(
        self,
        librispeech_dir: str,
        splits: list[str] | None = None,
        text_transform: TextTransform | None = None,
    ):
        if splits is None:
            splits = ['train-clean-100']

        self.text_transform = text_transform or build_text_transform()
        self.processor = Wav2Vec2Processor.from_pretrained(self.WAV2VEC2_MODEL)

        self.samples: list[tuple[str, str]] = []  # (audio_path, text)
        for split in splits:
            self._index_split(librispeech_dir, split)

    def _index_split(self, root: str, split: str) -> None:
        split_dir = os.path.join(root, 'LibriSpeech', split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f'LibriSpeech split not found: {split_dir}\n'
                f'Run scripts/download_data.sh first.'
            )

        for speaker_id in sorted(os.listdir(split_dir)):
            speaker_dir = os.path.join(split_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            for chapter_id in sorted(os.listdir(speaker_dir)):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                if not os.path.isdir(chapter_dir):
                    continue
                trans_file = os.path.join(
                    chapter_dir, f'{speaker_id}-{chapter_id}.trans.txt'
                )
                if not os.path.isfile(trans_file):
                    continue
                with open(trans_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        utt_id, *words = line.split()
                        text = ' '.join(words)
                        flac = os.path.join(chapter_dir, utt_id + '.flac')
                        if os.path.isfile(flac):
                            self.samples.append((flac, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        audio_path, text = self.samples[idx]

        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != self.TARGET_SR:
            import torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.TARGET_SR)
            audio = audio_t.squeeze(0).numpy()

        proc_out = self.processor(
            audio, sampling_rate=self.TARGET_SR,
            return_tensors='pt', padding=False,
        )
        audio_features = proc_out.input_values.squeeze(0)   # (T_audio,)

        try:
            text_int = torch.tensor(
                self.text_transform.text_to_int(text), dtype=torch.long,
            )
        except ValueError:
            text_int = torch.zeros(1, dtype=torch.long)

        return {
            'audio_features':   audio_features,
            'text_int':         text_int,
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
