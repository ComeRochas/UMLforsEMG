"""
Precompute the bare-minimum EMG tensors needed for CTC training.

For every sample in Gaddy's data layout this script:
  1. loads ``<idx>_emg.npy`` (plus the ± 1 neighbours, used only as context for
     the zero-phase filters)
  2. applies notch harmonics (60 Hz × 1..7) + 2 Hz highpass drift-removal
  3. subsamples to 689.06 Hz
  4. normalizes: ``raw_emg / 20`` then ``50 * tanh(raw_emg / 50)``
  5. caps length at 6400 frames (``limit_length=True`` in Gaddy's code)
  6. encodes ``info['text']`` via the shared ``TextTransform`` (char vocab)

Anything else Gaddy's pipeline does per ``__getitem__`` (get_emg_features,
load_audio, read_phonemes, mfcc/emg normalizers, silent→voiced parallel
lookup, per-sample pinning, lru_cache) is **thrown away** by our models and
is skipped here — which is why live data loading was starving the GPU.

The output is three self-contained torch archives

    <out_dir>/train.pt
    <out_dir>/dev.pt
    <out_dir>/test.pt

each a dict with

    raw_emg  : list[Tensor (T_i, 8)] in fp16  (~1 GB total for the train split)
    text_int : list[Tensor (L_i,)]   in int64
    version  : 1

Runs on CPU only.  Parallelized with ``multiprocessing``.

Usage
-----
    python src/precompute_emg.py \
        --emg_data_dir /scratch/cr4206/data/emg_data/emg_data \
        --out_dir      /scratch/cr4206/data/emg_cache \
        --num_workers  8
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Keep BLAS single-threaded so num_workers processes don't oversubscribe cores.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.signal
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_utils import TextTransform  # pure-Python, no absl FLAGS at use-time


LIMIT_LENGTH_MAX_RAW = 6400   # 800 feature frames × 8 stride — matches limit_length=True


# ---------------------------------------------------------------------------
# Signal-processing primitives (extracted from read_emg.py; self-contained)
# ---------------------------------------------------------------------------

def _notch(signal, freq, fs):
    b, a = scipy.signal.iirnotch(freq, 30, fs)
    return scipy.signal.filtfilt(b, a, signal)


def _notch_harmonics(signal, freq, fs):
    for h in range(1, 8):
        signal = _notch(signal, freq * h, fs)
    return signal


def _remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def _subsample(signal, new_fs, old_fs):
    times = np.arange(len(signal)) / old_fs
    sample_times = np.arange(0, times[-1], 1 / new_fs)
    return np.interp(sample_times, times, signal)


def _apply_to_all(fn, arr, *args, **kwargs):
    return np.stack(
        [fn(arr[:, i], *args, **kwargs) for i in range(arr.shape[1])], axis=1
    )


def load_raw_emg(base_dir: str, idx: int, limit_length: bool = True) -> np.ndarray:
    """Return a (T_raw, 8) float32 array of preprocessed 689 Hz raw_emg."""
    raw = np.load(os.path.join(base_dir, f'{idx}_emg.npy'))
    before_path = os.path.join(base_dir, f'{idx - 1}_emg.npy')
    after_path  = os.path.join(base_dir, f'{idx + 1}_emg.npy')
    before = np.load(before_path) if os.path.exists(before_path) else np.zeros([0, raw.shape[1]])
    after  = np.load(after_path)  if os.path.exists(after_path)  else np.zeros([0, raw.shape[1]])

    x = np.concatenate([before, raw, after], axis=0)
    x = _apply_to_all(_notch_harmonics, x, 60, 1000)
    x = _apply_to_all(_remove_drift, x, 1000)
    x = x[before.shape[0] : x.shape[0] - after.shape[0], :]
    emg = _apply_to_all(_subsample, x, 689.06, 1000)

    emg = emg / 20.0
    emg = 50.0 * np.tanh(emg / 50.0)

    if limit_length and emg.shape[0] > LIMIT_LENGTH_MAX_RAW:
        emg = emg[:LIMIT_LENGTH_MAX_RAW, :]

    return emg.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------

_TT: TextTransform | None = None


def _init_worker() -> None:
    global _TT
    _TT = TextTransform()


def _process_sample(task):
    base_dir, idx, text = task
    try:
        emg = load_raw_emg(base_dir, idx, limit_length=True).astype(np.float16)
        text_int = np.asarray(_TT.text_to_int(text), dtype=np.int64)
        if text_int.shape[0] == 0 or emg.shape[0] < 16:
            return None
        return {'raw_emg': emg, 'text_int': text_int}
    except Exception as exc:
        return {'_error': f'{base_dir}/{idx}: {exc!r}'}


# ---------------------------------------------------------------------------
# Split partitioning — replicates read_emg.EMGDataset's train/dev/test rule
# ---------------------------------------------------------------------------

def build_sample_manifest(emg_data_dir: str, testset_file: str) -> dict:
    """
    Return {'train': [...], 'dev': [...], 'test': [...]} where each entry is a
    (base_dir, idx, text) task tuple.

    Partition rule (matches read_emg.EMGDataset):
      * silent sessions:   included in train, dev and test
      * voiced sessions:   included in train only (excluded from dev/test
                           when any silent session is present)
    """
    with open(testset_file) as fh:
        tjson = json.load(fh)
    devset_s  = {tuple(x) for x in tjson['dev']}
    testset_s = {tuple(x) for x in tjson['test']}

    silent_root  = os.path.join(emg_data_dir, 'silent_parallel_data')
    voiced_roots = [
        os.path.join(emg_data_dir, 'voiced_parallel_data'),
        os.path.join(emg_data_dir, 'nonparallel_data'),
    ]
    has_silent = os.path.isdir(silent_root)

    session_dirs = []
    if has_silent:
        for s in sorted(os.listdir(silent_root)):
            p = os.path.join(silent_root, s)
            if os.path.isdir(p):
                session_dirs.append({'path': p, 'exclude_from_testset': False})
    for root in voiced_roots:
        if not os.path.isdir(root):
            continue
        for s in sorted(os.listdir(root)):
            p = os.path.join(root, s)
            if os.path.isdir(p):
                session_dirs.append({
                    'path': p,
                    'exclude_from_testset': has_silent,
                })

    out = {'train': [], 'dev': [], 'test': []}
    for d in session_dirs:
        for fname in sorted(os.listdir(d['path'])):
            m = re.match(r'(\d+)_info\.json', fname)
            if not m:
                continue
            idx = int(m.group(1))
            with open(os.path.join(d['path'], fname)) as fh:
                info = json.load(fh)
            if info.get('sentence_index', -1) < 0:
                continue
            loc = (info['book'], info['sentence_index'])
            task = (d['path'], idx, info['text'])
            in_test = loc in testset_s
            in_dev  = loc in devset_s
            if in_test and not d['exclude_from_testset']:
                out['test'].append(task)
            elif in_dev and not d['exclude_from_testset']:
                out['dev'].append(task)
            elif not in_test and not in_dev:
                out['train'].append(task)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--emg_data_dir', required=True,
                        help='Root containing silent_parallel_data/, voiced_parallel_data/, nonparallel_data/')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory — will contain train.pt, dev.pt, test.pt')
    parser.add_argument('--testset_file',
                        default=str(Path(_PROJECT_ROOT) / 'testset_largedev.json'),
                        help='JSON with dev/test split definitions')
    parser.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='Parallel CPU workers (default: #cores - 1)')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'],
                        choices=['train', 'dev', 'test'],
                        help='Which splits to materialize (default: all three)')
    args = parser.parse_args()

    t0 = time.time()
    print(f'[precompute_emg] emg_data_dir = {args.emg_data_dir}', flush=True)
    print(f'[precompute_emg] out_dir      = {args.out_dir}',      flush=True)
    print(f'[precompute_emg] num_workers  = {args.num_workers}',  flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print('[precompute_emg] scanning session directories ...', flush=True)
    manifest = build_sample_manifest(args.emg_data_dir, args.testset_file)
    for name in ('train', 'dev', 'test'):
        print(f'[precompute_emg]   split {name:5s}: {len(manifest[name])} samples', flush=True)

    for split in args.splits:
        tasks = manifest[split]
        if not tasks:
            print(f'[precompute_emg] skipping {split}: no samples', flush=True)
            continue

        print(
            f'[precompute_emg] === split={split} '
            f'({len(tasks)} samples, {args.num_workers} workers) ===',
            flush=True,
        )
        split_t0 = time.time()
        raw_emg_list: list[torch.Tensor] = []
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
                    raw_emg_list.append(torch.from_numpy(result['raw_emg']))
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
            'raw_emg':  raw_emg_list,
            'text_int': text_int_list,
            'version':  1,
        }
        print(f'[precompute_emg] saving {len(raw_emg_list)} samples → {out_path}', flush=True)
        torch.save(payload, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(
            f'[precompute_emg] {split}: wrote {len(raw_emg_list)} samples '
            f'({size_mb:.1f} MB) in {time.time() - split_t0:.1f}s '
            f'(errors={n_errors})',
            flush=True,
        )

    print(f'[precompute_emg] all done in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
