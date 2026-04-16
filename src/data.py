"""
Dataset classes for the silent-speech UML project.

EMGCharDataset   — wraps Gaddy's read_emg.py / EMGDataset.
                   On first use, builds a .pt cache with multiprocessing
                   (skipping MFCCs/phonemes we don't need) so subsequent
                   epochs load in ~2s instead of ~1h.
LibriSpeechCharDataset — loads LibriSpeech + normalizes waveforms via
                         Wav2Vec2Processor (feature extractor only).
                         Both datasets share the same TextTransform / vocab.
"""
import os
import sys
import multiprocessing as mp
import time
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
# Parallel cache builder — module-level so multiprocessing can pickle them
# ---------------------------------------------------------------------------

# Mutable dict filled once per worker process by the initializer.
_g_worker_state: dict = {}


def _cache_worker_init(limit_length: bool, text_align_directory: str,
                        project_root: str) -> None:
    """Called once per worker process before any tasks are dispatched."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Ensure absl FLAGS are parsed in the worker (inherited via fork, but
    # guard against the case where the pool uses 'spawn').
    from absl import flags as _flags
    _FLAGS = _flags.FLAGS
    if not _FLAGS.is_parsed():
        _FLAGS.mark_as_parsed()
    _FLAGS['remove_channels'].value = []

    from data_utils import TextTransform as _TT
    _g_worker_state['limit_length'] = limit_length
    _g_worker_state['text_align_directory'] = text_align_directory
    _g_worker_state['text_transform'] = _TT()


def _cache_worker(args: tuple) -> dict | None:
    """
    Load + DSP one EMG sample; return only raw_emg + text_int.

    Skips MFCCs, phonemes, and voiced features — none of which
    EMGCharDataset uses — so this is about 2× faster than the full
    EMGDataset.__getitem__ path.
    """
    from read_emg import load_utterance as _load

    directory, file_idx = args
    state = _g_worker_state

    try:
        _, _, text, _, _, raw_emg = _load(
            directory, file_idx,
            limit_length=state['limit_length'],
            text_align_directory=state['text_align_directory'],
        )
    except Exception as exc:
        print(f'[cache-worker] WARNING: {directory}/{file_idx} failed: {exc}',
              flush=True)
        return None

    # Same normalisation as EMGDataset.__getitem__
    raw_emg = raw_emg / 20.0
    raw_emg = (50.0 * np.tanh(raw_emg / 50.0)).astype(np.float32)

    try:
        text_int = np.array(
            state['text_transform'].text_to_int(text), dtype=np.int64
        )
    except ValueError:
        return None

    return {
        'raw_emg': torch.from_numpy(raw_emg),
        'text_int': torch.from_numpy(text_int),
    }


# ---------------------------------------------------------------------------
# EMGCharDataset
# ---------------------------------------------------------------------------

def _init_emg_flags(emg_data_dir: str, normalizers_file: str,
                    testset_file: str, text_align_directory: str) -> None:
    """
    Set absl FLAGS used by read_emg.py / data_utils.py.
    Must be called before instantiating EMGDataset.
    """
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
    Wraps Gaddy's EMGDataset.  Only raw_emg + text_int are returned.

    Pass cache_path to enable the fast path:
      • First run  — builds a .pt cache in parallel (~8 min with 8 workers).
      • Later runs — loads the cache in ~2s; DataLoader uses num_workers=0.

    Without cache_path the dataset falls back to the slow per-sample path
    (lru_cache in EMGDataset helps only in single-process mode).

    Returns per-sample dicts with keys:
        raw_emg          : FloatTensor (T_raw, 8)
        text_int         : LongTensor  (L,)
        lengths          : int   — T_raw
        text_int_lengths : int   — L
    """

    def __init__(
        self,
        emg_data_dir: str,
        split: str = 'train',
        normalizers_file: str | None = None,
        testset_file: str | None = None,
        text_align_directory: str | None = None,
        cache_path: str | None = None,
        index_cache_path: str | None = None,
        num_cache_workers: int = 8,
    ):
        root = Path(emg_data_dir).parent
        if normalizers_file is None:
            normalizers_file = str(root / 'normalizers.pkl')
        if testset_file is None:
            testset_file = str(Path(_PROJECT_ROOT) / 'testset_largedev.json')
        if text_align_directory is None:
            text_align_directory = str(Path(_PROJECT_ROOT) / 'text_alignments')
        if index_cache_path is None:
            index_cache_path = str(root / 'emg_cache' / f'index_{split}.pkl')

        _init_emg_flags(
            emg_data_dir=emg_data_dir,
            normalizers_file=normalizers_file,
            testset_file=testset_file,
            text_align_directory=text_align_directory,
        )

        is_dev  = (split == 'dev')
        is_test = (split == 'test')
        print(
            f'[data] creating EMGDataset split={split} dev={is_dev} test={is_test} ...',
            flush=True,
        )
        t_inner = time.time()
        self._inner = EMGDataset(
            limit_length=True,
            dev=is_dev,
            test=is_test,
            index_cache_path=index_cache_path,
        )
        print(
            f'[data] EMGDataset ready in {time.time() - t_inner:.2f}s '
            f'with {len(self._inner)} items',
            flush=True,
        )
        self.text_transform = self._inner.text_transform

        # ------------------------------------------------------------------
        # Build or load cache
        # ------------------------------------------------------------------
        if cache_path is not None:
            cp = Path(cache_path)
            if cp.exists():
                print(f'[data] loading cache {cp} ...', flush=True)
                self._cache: list[dict] | None = torch.load(cp, weights_only=False)
                print(f'[data] cache loaded — {len(self._cache)} samples', flush=True)
            else:
                print(
                    f'[data] cache not found — building with {num_cache_workers} workers '
                    f'({len(self._inner)} samples) ...', flush=True
                )
                self._cache = self._build_cache(
                    text_align_directory, num_cache_workers
                )
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._cache, cp)
                print(f'[data] cache saved → {cp}', flush=True)
        else:
            self._cache = None

    # ------------------------------------------------------------------
    # Cache builder
    # ------------------------------------------------------------------

    def _build_cache(self, text_align_directory: str,
                     n_workers: int) -> list[dict]:
        args_list = [
            (info.directory, int(file_idx))
            for info, file_idx in self._inner.example_indices
        ]
        init_args = (
            self._inner.limit_length,
            text_align_directory,
            _PROJECT_ROOT,
        )
        n_total = len(args_list)
        results = []

        # Tuning knobs for diagnostics/perf on shared clusters.
        progress_every = max(1, int(os.environ.get('EMG_CACHE_PROGRESS_EVERY', '20')))
        chunksize = max(1, int(os.environ.get('EMG_CACHE_CHUNKSIZE', '8')))
        use_unordered = os.environ.get('EMG_CACHE_UNORDERED', '1').lower() not in ('0', 'false', 'no')
        print(
            f'[data] cache config: workers={n_workers} chunksize={chunksize} '
            f'unordered={use_unordered} progress_every={progress_every}',
            flush=True,
        )

        started = time.time()

        if n_workers <= 0:
            # ---- Sequential mode — avoids fork/spawn deadlocks ----
            print('[data] cache build: sequential mode (workers=0)', flush=True)
            _cache_worker_init(*init_args)
            for i, args in enumerate(args_list, start=1):
                result = _cache_worker(args)
                results.append(result)
                if i % progress_every == 0 or i == n_total:
                    elapsed = time.time() - started
                    pct = 100 * i / n_total
                    rate = i / max(elapsed, 1e-9)
                    eta_s = (n_total - i) / max(rate, 1e-9)
                    print(
                        f'[data] cache build: {i}/{n_total} ({pct:.0f}%) '
                        f'elapsed={elapsed:.1f}s rate={rate:.2f} ex/s eta={eta_s:.1f}s',
                        flush=True,
                    )
        else:
            # ---- Parallel mode ----
            # Use 'spawn' — 'fork' deadlocks after torch/numpy import
            # because forked children inherit thread-pool locks without
            # the threads that hold them.
            ctx = mp.get_context('spawn')
            print(
                f'[data] cache build: parallel mode (spawn, {n_workers} workers)',
                flush=True,
            )
            with ctx.Pool(
                processes=n_workers,
                initializer=_cache_worker_init,
                initargs=init_args,
            ) as pool:
                iterator = pool.imap_unordered if use_unordered else pool.imap
                for i, result in enumerate(
                    iterator(_cache_worker, args_list, chunksize=chunksize),
                    start=1,
                ):
                    results.append(result)
                    if i % progress_every == 0 or i == n_total:
                        elapsed = time.time() - started
                        pct = 100 * i / n_total
                        rate = i / max(elapsed, 1e-9)
                        eta_s = (n_total - i) / max(rate, 1e-9)
                        print(
                            f'[data] cache build: {i}/{n_total} ({pct:.0f}%) '
                            f'elapsed={elapsed:.1f}s rate={rate:.2f} ex/s '
                            f'eta={eta_s:.1f}s',
                            flush=True,
                        )

        cache = [r for r in results if r is not None]
        n_failed = n_total - len(cache)
        if n_failed:
            print(f'[data] WARNING: {n_failed} samples failed and were skipped.',
                  flush=True)
        return cache

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._cache is not None:
            return len(self._cache)
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            sample = self._cache[idx]
            raw_emg  = sample['raw_emg']
            text_int = sample['text_int']
        else:
            inner    = self._inner[idx]
            raw_emg  = inner['raw_emg'].float()
            text_int = inner['text_int'].long()

        T_raw = raw_emg.shape[0]
        L     = text_int.shape[0]
        return {
            'raw_emg':          raw_emg,
            'text_int':         text_int,
            'lengths':          T_raw,
            'text_int_lengths': L,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Pad variable-length sequences and stack into batch tensors."""
        from torch.nn.utils.rnn import pad_sequence

        raw_emg_list = [b['raw_emg'] for b in batch]
        text_list    = [b['text_int'] for b in batch]
        lengths      = torch.tensor([b['lengths'] for b in batch])
        text_lengths = torch.tensor([b['text_int_lengths'] for b in batch])

        raw_emg_padded = pad_sequence(raw_emg_list, batch_first=True, padding_value=0.0)
        text_padded    = pad_sequence(text_list,    batch_first=True, padding_value=0)

        return {
            'raw_emg':          raw_emg_padded,
            'text_int':         text_padded,
            'lengths':          lengths,
            'text_int_lengths': text_lengths,
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

        self.samples: list[tuple[str, str]] = []
        for split in splits:
            self._index_split(librispeech_dir, split)

    def _index_split(self, root: str, split: str) -> None:
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
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors='pt',
            padding=False,
        )
        audio_features = proc_out.input_values.squeeze(0)

        try:
            text_int = torch.tensor(
                self.text_transform.text_to_int(text), dtype=torch.long
            )
        except ValueError:
            text_int = torch.zeros(1, dtype=torch.long)

        L = text_int.shape[0]
        return {
            'audio_features':   audio_features,
            'text_int':         text_int,
            'text_int_lengths': L,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        from torch.nn.utils.rnn import pad_sequence

        audio_list    = [b['audio_features'] for b in batch]
        text_list     = [b['text_int']        for b in batch]
        audio_lengths = torch.tensor([b['audio_features'].shape[0] for b in batch])
        text_lengths  = torch.tensor([b['text_int_lengths']         for b in batch])

        audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        text_padded  = pad_sequence(text_list,  batch_first=True, padding_value=0)

        return {
            'audio_features':   audio_padded,
            'audio_lengths':    audio_lengths,
            'text_int':         text_padded,
            'text_int_lengths': text_lengths,
        }


# ---------------------------------------------------------------------------
# LibriSpeechFeatureDataset
# ---------------------------------------------------------------------------

class LibriSpeechFeatureDataset(Dataset):
    """
    Loads precomputed wav2vec2-base features from an on-disk cache.

    Each utterance is stored as a .pt file containing:
        features : Tensor(T', 768) fp16 — last_hidden_state of frozen wav2vec2
        text_int : Tensor(L,)      int64

    The cache is produced by src/precompute_librispeech_features.py.  At
    training time the (frozen) wav2vec2 forward pass is skipped entirely —
    only the trainable projection + SharedTransformer run on the GPU.

    Returns per-sample dicts with keys:
        features         : FloatTensor (T', 768)
        text_int         : LongTensor  (L,)
        feat_lengths     : int         — T'
        text_int_lengths : int         — L
    """

    INDEX_NAME = 'index.pt'

    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)
        index_path = self.features_dir / self.INDEX_NAME
        if not index_path.is_file():
            raise FileNotFoundError(
                f'Feature-cache index not found: {index_path}\n'
                f'Run src/precompute_librispeech_features.py first.'
            )
        self.files: list[str] = torch.load(index_path, weights_only=False)
        print(
            f'[data] LibriSpeechFeatureDataset: {len(self.files)} utterances '
            f'from {self.features_dir}',
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        rec = torch.load(self.features_dir / self.files[idx], weights_only=False)
        features = rec['features'].float()        # fp16 on disk → fp32 in RAM
        text_int = rec['text_int'].long()
        return {
            'features':         features,
            'text_int':         text_int,
            'feat_lengths':     features.shape[0],
            'text_int_lengths': text_int.shape[0],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        from torch.nn.utils.rnn import pad_sequence

        feat_list    = [b['features'] for b in batch]
        text_list    = [b['text_int'] for b in batch]
        feat_lengths = torch.tensor([b['feat_lengths']     for b in batch])
        text_lengths = torch.tensor([b['text_int_lengths'] for b in batch])

        feat_padded = pad_sequence(feat_list, batch_first=True, padding_value=0.0)
        text_padded = pad_sequence(text_list, batch_first=True, padding_value=0)

        return {
            'features':         feat_padded,
            'feat_lengths':     feat_lengths,
            'text_int':         text_padded,
            'text_int_lengths': text_lengths,
        }
