"""
Precompute the EMG .pt cache for train + dev splits.

EMGCharDataset builds the cache automatically on first access — this script
is a thin wrapper so the cache is produced by a dedicated (CPU-bound) job,
separate from GPU training time.

No GPU required — run directly from the login node or any terminal:

    python src/precompute_emg_cache.py \\
        --emg_data_dir  /scratch/cr4206/data/emg_data/emg_data \\
        --emg_cache_dir /scratch/cr4206/data/emg_cache \\
        [--splits train dev] \\
        [--num_workers 0]        # 0 = sequential (default, no deadlock risk)
"""
import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import EMGCharDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--emg_data_dir',  required=True)
    parser.add_argument('--emg_cache_dir', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'dev'])
    parser.add_argument('--num_workers', type=int, default=0,
                        help='0 = sequential (safe), >0 = spawn pool (faster)')
    args = parser.parse_args()

    Path(args.emg_cache_dir).mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        cache_path = os.path.join(args.emg_cache_dir, f'{split}.pt')
        print(f'\n[emg-cache] ===== split={split} → {cache_path} =====', flush=True)
        t0 = time.time()
        dset = EMGCharDataset(
            emg_data_dir=args.emg_data_dir,
            split=split,
            cache_path=cache_path,
            num_cache_workers=args.num_workers,
        )
        print(
            f'[emg-cache] split={split} done  items={len(dset)}  '
            f'elapsed={time.time()-t0:.1f}s',
            flush=True,
        )

    print('\n[emg-cache] all splits done.', flush=True)


if __name__ == '__main__':
    main()
