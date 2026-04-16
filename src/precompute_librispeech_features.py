"""
Precompute wav2vec2-base features for LibriSpeech, writing one .pt file per
utterance plus an index.pt so LibriSpeechFeatureDataset can load them fast.

The wav2vec2 encoder is frozen during UML training, so its output is
deterministic w.r.t. the input waveform — computing it once here removes
~95M frozen params from the UML training hot path.

Output layout:
    <features_dir>/
        <utt_id>.pt         — {'features': fp16 (T', 768), 'text_int': int64}
        index.pt            — list[str] of filenames (sorted, deterministic)

Usage:
    python src/precompute_librispeech_features.py \\
        --librispeech_dir /scratch/cr4206/data/librispeech \\
        --features_dir    /scratch/cr4206/data/librispeech_features \\
        [--splits train-clean-100] \\
        [--batch_size 16]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import LibriSpeechCharDataset, build_text_transform


def _utt_id_from_path(audio_path: str) -> str:
    """LibriSpeech filenames are unique (e.g. 103-1240-0000.flac)."""
    return Path(audio_path).stem


def _collate_with_utt_ids(batch: list[dict], samples_with_ids: list[tuple]) -> dict:
    """
    LibriSpeechCharDataset.collate_fn doesn't expose utt_ids — we need them
    to name the output files.  Work around by reusing the dataset's samples
    list when we index.
    """
    audio_list    = [b['audio_features'] for b in batch]
    text_list     = [b['text_int']        for b in batch]
    audio_lengths = torch.tensor([b['audio_features'].shape[0] for b in batch])
    text_lengths  = torch.tensor([b['text_int_lengths']         for b in batch])
    audio_padded  = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    text_padded   = pad_sequence(text_list,  batch_first=True, padding_value=0)
    return {
        'audio_features':   audio_padded,
        'audio_lengths':    audio_lengths,
        'text_int':         text_padded,
        'text_int_lengths': text_lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--librispeech_dir', required=True)
    parser.add_argument('--features_dir',    required=True)
    parser.add_argument('--splits', nargs='+', default=['train-clean-100'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[precompute] device={device} splits={args.splits}', flush=True)

    # ------------------------------------------------------------------
    # Load dataset (gives us processor-normalized waveforms)
    # ------------------------------------------------------------------
    t0 = time.time()
    text_transform = build_text_transform()
    dataset = LibriSpeechCharDataset(
        librispeech_dir=args.librispeech_dir,
        splits=args.splits,
        text_transform=text_transform,
    )
    n_total = len(dataset)
    print(f'[precompute] indexed {n_total} utterances in {time.time()-t0:.1f}s',
          flush=True)

    # Pre-extract utt_ids so we can name output files matching batch order.
    # We iterate the dataset via a SequentialSampler to keep (index → utt_id) stable.
    utt_ids = [_utt_id_from_path(audio_path) for audio_path, _ in dataset.samples]

    # ------------------------------------------------------------------
    # Skip utterances already cached (so this job is idempotent/resumable)
    # ------------------------------------------------------------------
    todo_indices = [
        i for i, uid in enumerate(utt_ids)
        if not (out_dir / f'{uid}.pt').is_file()
    ]
    n_done = n_total - len(todo_indices)
    print(
        f'[precompute] {n_done} already cached, {len(todo_indices)} to compute',
        flush=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    t0 = time.time()
    w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(device)
    w2v.eval()
    for p in w2v.parameters():
        p.requires_grad = False
    print(f'[precompute] wav2vec2 loaded in {time.time()-t0:.1f}s', flush=True)

    # ------------------------------------------------------------------
    # Batched forward with manual indexing so we know which utt_id each row is
    # ------------------------------------------------------------------
    batch_size = args.batch_size
    n_todo = len(todo_indices)
    started = time.time()
    log_every = max(1, n_todo // 50)

    with torch.inference_mode():
        for b_start in range(0, n_todo, batch_size):
            b_indices = todo_indices[b_start : b_start + batch_size]

            samples = [dataset[i] for i in b_indices]
            audio_list = [s['audio_features'] for s in samples]
            text_list  = [s['text_int']        for s in samples]
            audio_lens = torch.tensor([a.shape[0] for a in audio_list])

            audio_padded = pad_sequence(audio_list, batch_first=True,
                                         padding_value=0.0).to(device)
            attention_mask = (
                torch.arange(audio_padded.shape[1], device=device).unsqueeze(0)
                < audio_lens.to(device).unsqueeze(1)
            ).long()

            out = w2v(input_values=audio_padded, attention_mask=attention_mask)
            features = out.last_hidden_state                    # (B, T', 768)

            # Output-frame counts
            if hasattr(w2v, '_get_feat_extract_output_lengths'):
                feat_lens = w2v._get_feat_extract_output_lengths(audio_lens.to(device))
            else:
                feat_lens = (audio_lens.to(device) - 400) // 320 + 1
            feat_lens = feat_lens.long().clamp(min=1, max=features.shape[1])

            # Save one .pt per utterance — no padding, fp16
            for i, idx in enumerate(b_indices):
                T_i = int(feat_lens[i].item())
                rec = {
                    'features': features[i, :T_i].to(torch.float16).cpu(),
                    'text_int': text_list[i].long(),
                }
                torch.save(rec, out_dir / f'{utt_ids[idx]}.pt')

            done_now = b_start + len(b_indices)
            if (b_start // batch_size) % log_every == 0 or done_now == n_todo:
                elapsed = time.time() - started
                rate = done_now / max(elapsed, 1e-9)
                eta_s = (n_todo - done_now) / max(rate, 1e-9)
                print(
                    f'[precompute] {done_now}/{n_todo} '
                    f'({100*done_now/max(n_todo,1):.0f}%) '
                    f'elapsed={elapsed:.1f}s rate={rate:.2f} utt/s '
                    f'eta={eta_s:.0f}s',
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Write index (sorted list of {utt_id}.pt filenames that actually exist)
    # ------------------------------------------------------------------
    all_files = sorted(
        p.name for p in out_dir.glob('*.pt') if p.name != 'index.pt'
    )
    torch.save(all_files, out_dir / 'index.pt')
    print(f'[precompute] wrote index.pt with {len(all_files)} entries '
          f'→ {out_dir}/index.pt', flush=True)
    print('[precompute] done.', flush=True)


if __name__ == '__main__':
    main()
