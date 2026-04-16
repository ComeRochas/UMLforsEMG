"""
Dataset classes for the silent-speech UML project.

EMGCharDataset   — wraps Gaddy's read_emg.py / EMGDataset
LibriSpeechCharDataset — loads LibriSpeech + normalizes waveforms via
                         Wav2Vec2Processor (feature extractor only).
                         Both datasets share the same TextTransform / vocab.
"""
import os
import sys
import string
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import read_emg / data_utils
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from absl import flags

# Import Gaddy modules (they register absl FLAGS as a side-effect)
import read_emg as _read_emg_module  # noqa: F401 — registers FLAGS
import data_utils as _data_utils_module  # noqa: F401 — registers FLAGS

from read_emg import EMGDataset
from data_utils import TextTransform

FLAGS = flags.FLAGS


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
# EMGCharDataset
# ---------------------------------------------------------------------------

def _init_emg_flags(emg_data_dir: str, normalizers_file: str,
                    testset_file: str, text_align_directory: str) -> None:
    """
    Set absl FLAGS used by read_emg.py / data_utils.py.
    Must be called before instantiating EMGDataset.
    """
    # Mark parsed if not already done (avoids absl "not parsed" errors).
    if not FLAGS.is_parsed():
        FLAGS.mark_as_parsed()

    FLAGS['silent_data_directories'].value = [
        os.path.join(emg_data_dir, 'silent_parallel_data'),
    ]
    FLAGS['voiced_data_directories'].value = [
        os.path.join(emg_data_dir, 'voiced_parallel_data'),
        os.path.join(emg_data_dir, 'nonparallel_data'),
    ]
    FLAGS['testset_file'].value = testset_file
    FLAGS['text_align_directory'].value = text_align_directory
    FLAGS['normalizers_file'].value = normalizers_file
    FLAGS['remove_channels'].value = []


class EMGCharDataset(Dataset):
    """
    Wraps Gaddy's EMGDataset.  Only raw_emg is returned (the hand-crafted EMG
    feature matrix is not needed; the CNN operates directly on raw EMG).

    Returns per-sample dicts with keys:

        raw_emg          : FloatTensor (T_raw, 8)  — raw 8-ch EMG
                           T_raw ≈ 8 × T_features; after 3×stride-2 CNN → T_features
        text_int         : LongTensor  (L,)
        lengths          : int   — T_raw (used to compute CTC input length = T_raw // 8)
        text_int_lengths : int   — L
    """

    def __init__(
        self,
        emg_data_dir: str,
        split: str = 'train',           # 'train' | 'dev' | 'test'
        normalizers_file: str | None = None,
        testset_file: str | None = None,
        text_align_directory: str | None = None,
    ):
        # Resolve defaults relative to emg_data_dir parent
        root = Path(emg_data_dir).parent
        if normalizers_file is None:
            normalizers_file = str(root / 'normalizers.pkl')
        if testset_file is None:
            testset_file = str(Path(_PROJECT_ROOT) / 'testset_largedev.json')
        if text_align_directory is None:
            text_align_directory = str(Path(_PROJECT_ROOT) / 'text_alignments')

        _init_emg_flags(
            emg_data_dir=emg_data_dir,
            normalizers_file=normalizers_file,
            testset_file=testset_file,
            text_align_directory=text_align_directory,
        )

        is_dev  = (split == 'dev')
        is_test = (split == 'test')
        self._inner = EMGDataset(
            limit_length=True,
            dev=is_dev,
            test=is_test,
        )
        self.text_transform = self._inner.text_transform

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self._inner[idx]

        raw_emg  = sample['raw_emg']   # FloatTensor (T_raw, 8)
        text_int = sample['text_int']  # LongTensor  (L,)

        T_raw = raw_emg.shape[0]
        L     = text_int.shape[0]

        return {
            'raw_emg':          raw_emg.float(),
            'text_int':         text_int.long(),
            'lengths':          T_raw,   # T_raw; model computes enc_len = T_raw // 8
            'text_int_lengths': L,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Pad variable-length sequences and stack into batch tensors."""
        from torch.nn.utils.rnn import pad_sequence

        raw_emg_list = [b['raw_emg'] for b in batch]   # each (T_raw_i, 8)
        text_list    = [b['text_int'] for b in batch]  # each (L_i,)
        lengths      = torch.tensor([b['lengths'] for b in batch])           # (B,) T_raw per sample
        text_lengths = torch.tensor([b['text_int_lengths'] for b in batch])  # (B,)

        # pad_sequence expects list of (T_i, ...) tensors; batch_first=True → (B, T_max, 8)
        raw_emg_padded = pad_sequence(raw_emg_list, batch_first=True, padding_value=0.0)
        text_padded    = pad_sequence(text_list,    batch_first=True, padding_value=0)

        return {
            'raw_emg':          raw_emg_padded,  # (B, T_raw_max, 8)
            'text_int':         text_padded,      # (B, L_max)
            'lengths':          lengths,           # (B,) T_raw
            'text_int_lengths': text_lengths,      # (B,)
        }


# ---------------------------------------------------------------------------
# LibriSpeechCharDataset
# ---------------------------------------------------------------------------

class LibriSpeechCharDataset(Dataset):
    """
    Loads LibriSpeech audio + transcriptions.

    The raw waveform is normalized by Wav2Vec2Processor (feature extractor
    only – no model forward pass here).  The wav2vec2 encoder is applied
    inside AudioEncoder at training time.

    Returns per-sample dicts with keys:

        audio_features   : FloatTensor (T_audio,)  — normalized waveform
        text_int         : LongTensor  (L,)
        text_int_lengths : int
    """

    # LibriSpeech book-level directory layout:
    # <root>/LibriSpeech/<split>/<speaker>/<chapter>/<utt>.flac
    # Transcriptions: <root>/LibriSpeech/<split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

    WAV2VEC2_MODEL = 'facebook/wav2vec2-base'
    TARGET_SR      = 16_000   # wav2vec2 expects 16 kHz

    def __init__(
        self,
        librispeech_dir: str,
        splits: list[str] | None = None,
        text_transform: TextTransform | None = None,
    ):
        """
        Args:
            librispeech_dir: root of extracted LibriSpeech (contains
                             LibriSpeech/<split>/<speaker>/... structure).
            splits: list of split names, e.g. ['train-clean-100'].
                    Defaults to ['train-clean-100'].
            text_transform: shared TextTransform instance; created if None.
        """
        if splits is None:
            splits = ['train-clean-100']

        self.text_transform = text_transform or build_text_transform()
        self.processor = Wav2Vec2Processor.from_pretrained(self.WAV2VEC2_MODEL)

        self.samples: list[tuple[str, str]] = []  # (audio_path, text)
        for split in splits:
            self._index_split(librispeech_dir, split)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_split(self, root: str, split: str) -> None:
        """Walk the split directory and collect (flac_path, transcript) pairs."""
        split_dir = os.path.join(root, 'LibriSpeech', split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"LibriSpeech split not found: {split_dir}\n"
                f"Run scripts/download_data.sh first."
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

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        audio_path, text = self.samples[idx]

        # Load audio (LibriSpeech is 16 kHz)
        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != self.TARGET_SR:
            import torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.TARGET_SR)
            audio = audio_t.squeeze(0).numpy()

        # Normalize via wav2vec2 feature extractor (no model forward)
        proc_out = self.processor(
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors='pt',
            padding=False,
        )
        # input_values: (1, T_audio) → (T_audio,)
        audio_features = proc_out.input_values.squeeze(0)  # FloatTensor

        # Encode text using SAME vocabulary as EMGCharDataset
        try:
            text_int = torch.tensor(
                self.text_transform.text_to_int(text), dtype=torch.long
            )
        except ValueError:
            # Unknown character — skip by returning empty (handled in collate)
            text_int = torch.zeros(1, dtype=torch.long)

        L = text_int.shape[0]
        return {
            'audio_features':   audio_features,   # (T_audio,)
            'text_int':         text_int,          # (L,)
            'text_int_lengths': L,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Pad waveforms and text sequences; return padded batch tensors.

        Returns audio_lengths so that AudioEncoder can build an attention_mask
        and pass it to wav2vec2, preventing the model from attending to
        zero-padded frames.
        """
        from torch.nn.utils.rnn import pad_sequence

        audio_list   = [b['audio_features'] for b in batch]  # each (T_audio_i,)
        text_list    = [b['text_int']        for b in batch]  # each (L_i,)
        audio_lengths = torch.tensor([b['audio_features'].shape[0] for b in batch])  # (B,)
        text_lengths  = torch.tensor([b['text_int_lengths']         for b in batch])  # (B,)

        # pad_sequence expects (T_i,) tensors → stacks to (B, T_max) with batch_first
        audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        text_padded  = pad_sequence(text_list,  batch_first=True, padding_value=0)

        return {
            'audio_features':   audio_padded,    # (B, T_audio_max)
            'audio_lengths':    audio_lengths,    # (B,)  — real sample lengths
            'text_int':         text_padded,      # (B, L_max)
            'text_int_lengths': text_lengths,     # (B,)
        }
