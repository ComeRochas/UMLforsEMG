#!/usr/bin/env python3
import argparse
import os
import socket
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg-data-dir", default="/scratch/cr4206/data/emg_data/emg_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-path", default="/scratch/cr4206/data/emg_cache/train.pt")
    parser.add_argument("--index-cache-path", default=None)
    args = parser.parse_args()

    print(f"[diag] host={socket.gethostname()} pid={os.getpid()} time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[diag] cwd={os.getcwd()}", flush=True)
    print(f"[diag] emg_data_dir={args.emg_data_dir}", flush=True)
    print(f"[diag] split={args.split}", flush=True)
    print(f"[diag] cache_path={args.cache_path}", flush=True)
    print(f"[diag] index_cache_path={args.index_cache_path}", flush=True)

    try:
        t0 = time.time()
        print("[diag] importing EMGCharDataset...", flush=True)
        from src.data import EMGCharDataset
        print(f"[diag] import done in {time.time() - t0:.2f}s", flush=True)

        t1 = time.time()
        print("[diag] constructing dataset...", flush=True)
        dset = EMGCharDataset(
            emg_data_dir=args.emg_data_dir,
            split=args.split,
            cache_path=args.cache_path,
            index_cache_path=args.index_cache_path,
        )
        print(f"[diag] dataset constructed in {time.time() - t1:.2f}s", flush=True)
        print(f"[diag] dataset length={len(dset)}", flush=True)

        t2 = time.time()
        print("[diag] reading first sample...", flush=True)
        sample = dset[0]
        print(f"[diag] first sample fetched in {time.time() - t2:.2f}s", flush=True)
        print(f"[diag] sample keys={list(sample.keys())}", flush=True)
        print(f"[diag] raw_emg shape={tuple(sample['raw_emg'].shape)}", flush=True)
        print(f"[diag] text_int length={int(sample['text_int'].shape[0])}", flush=True)

        print("[diag] success", flush=True)
    except Exception:
        print("[diag] exception raised:", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
