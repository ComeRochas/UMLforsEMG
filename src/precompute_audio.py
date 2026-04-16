"""
Precompute LibriSpeech waveforms + char-level text_int for the UML audio branch.

Per sample this script:
  1. decodes FLAC → fp32 (T,) via soundfile
  2. (if needed) resamples to 16 kHz — LibriSpeech is already 16 kHz so this
     branch is usually dead
  3. applies zero-mean / unit-variance normalization per-sample
     ``(x - x.mean()) / sqrt(x.var() + 1e-7)`` — identical to
     ``transformers.Wav2Vec2FeatureExtractor`` defaults, so we can drop
     Wav2Vec2Processor from the runtime path
  4. casts to fp16
  5. encodes the transcript via the shared ``TextTransform`` (char vocab)

Writes one file per split to ``<out_dir>/<split>.pt`` — each a dict:

    audio    : list[Tensor (T_i,)]  — fp16 normalized waveforms
    text_int : list[Tensor (L_i,)]  — int64 char-level token ids
    version  : 1

Runs on CPU only.  Parallelized with ``ProcessPoolExecutor``.

Usage
-----
    python src/precompute_audio.py \\
        --librispeech_dir /scratch/cr4206/data/librispeech \\
        --out_dir         /scratch/cr4206/data/libri_cache \\
        --splits          train-clean-100 \\
        --num_workers     8
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Keep BLAS single-threaded so num_workers processes don't oversubscribe cores.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_utils import TextTransform


TARGET_SR = 16_000
EPS       = 1e-7


# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------

_TT: TextTransform | None = None


def _init_worker() -> None:
    global _TT
    _TT = TextTransform()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _zero_mean_unit_var(x: np.ndarray) -> np.ndarray:
    """Match Wav2Vec2FeatureExtractor._zero_mean_unit_var_norm (per-sample)."""
    return (x - x.mean()) / np.sqrt(x.var() + EPS)


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    import torchaudio
    t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, TARGET_SR)
    return t.squeeze(0).numpy()


def _process_sample(task):
    audio_path, text = task
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = _resample(audio, sr)
        audio = _zero_mean_unit_var(audio).astype(np.float16)
        try:
            text_int = np.asarray(_TT.text_to_int(text), dtype=np.int64)
        except ValueError:
            return None
        if text_int.shape[0] == 0 or audio.shape[0] < 16:
            return None
        return {'audio': audio, 'text_int': text_int}
    except Exception as exc:
        return {'_error': f'{audio_path}: {exc!r}'}


# ---------------------------------------------------------------------------
# Split manifest (mirrors LibriSpeechCharDataset._index_split)
# ---------------------------------------------------------------------------

def build_manifest(root: str, split: str) -> list[tuple[str, str]]:
    split_dir = os.path.join(root, 'LibriSpeech', split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f'LibriSpeech split not found: {split_dir}\n'
            f'Run scripts/download_data.sh first.'
        )
    tasks: list[tuple[str, str]] = []
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
                    flac = os.path.join(chapter_dir, utt_id + '.flac')
                    if os.path.isfile(flac):
                        tasks.append((flac, ' '.join(words)))
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--librispeech_dir', required=True,
                        help='Root containing LibriSpeech/<split>/<speaker>/<chapter>/*.flac')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory — one <split>.pt file per split')
    parser.add_argument('--splits', nargs='+', default=['train-clean-100'],
                        help='LibriSpeech splits to materialize (default: train-clean-100)')
    parser.add_argument('--num_workers', type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help='Parallel CPU workers (default: #cores - 1)')
    args = parser.parse_args()

    t0 = time.time()
    print(f'[precompute_audio] librispeech_dir = {args.librispeech_dir}', flush=True)
    print(f'[precompute_audio] out_dir         = {args.out_dir}',         flush=True)
    print(f'[precompute_audio] num_workers     = {args.num_workers}',     flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        print(f'[precompute_audio] scanning split={split} ...', flush=True)
        tasks = build_manifest(args.librispeech_dir, split)
        print(f'[precompute_audio]   {len(tasks)} samples', flush=True)
        if not tasks:
            continue

        split_t0 = time.time()
        audio_list:    list[torch.Tensor] = []
        text_int_list: list[torch.Tensor] = []
        n_errors = 0
        log_every = max(1, len(tasks) // 40)   # ~40 progress lines per split

        with ProcessPoolExecutor(
            max_workers=args.num_workers, initializer=_init_worker,
        ) as ex:
            for i, result in enumerate(
                ex.map(_process_sample, tasks, chunksize=16), start=1,
            ):
                if result is None:
                    pass
                elif '_error' in result:
                    n_errors += 1
                    if n_errors <= 5:
                        print(f'  [err] {result["_error"]}', flush=True)
                else:
                    audio_list.append(torch.from_numpy(result['audio']))
                    text_int_list.append(torch.from_numpy(result['text_int']))

                if i % log_every == 0 or i == len(tasks):
                    elapsed = time.time() - split_t0
                    rate    = i / max(1e-3, elapsed)
                    eta     = (len(tasks) - i) / max(rate, 1e-3)
                    print(
                        f'  [{split}] {i}/{len(tasks)} ({100*i/len(tasks):5.1f}%) '
                        f'rate={rate:6.1f} samp/s  elapsed={elapsed:6.1f}s  eta={eta:6.0f}s',
                        flush=True,
                    )

        out_path = os.path.join(args.out_dir, f'{split}.pt')
        payload = {
            'audio':    audio_list,
            'text_int': text_int_list,
            'version':  1,
        }
        print(f'[precompute_audio] saving {len(audio_list)} samples → {out_path}', flush=True)
        torch.save(payload, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(
            f'[precompute_audio] {split}: wrote {len(audio_list)} samples '
            f'({size_mb:.1f} MB) in {time.time() - split_t0:.1f}s '
            f'(errors={n_errors})',
            flush=True,
        )

    print(f'[precompute_audio] all done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
