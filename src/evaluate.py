"""
Evaluate a saved checkpoint (baseline, finetune, or uml) on the Gaddy test set.

Loads BaselineModel for all three run types (the EMG branch of UMLModel has the
same architecture as BaselineModel, and finetune_from_uml.py saves a
BaselineModel checkpoint).  Reports character-level WER on the test split.

Usage:
    python src/evaluate.py \
        --checkpoint /scratch/cr4206/checkpoints/baseline/best.pt \
        --config     configs/baseline.yaml \
        [--split     test]          # 'dev' or 'test' (default: test)
        [--beam_width 50]           # greedy if 0 (default)
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml
import jiwer

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import EMGCharDataset, build_text_transform, vocab_size, blank_id
from src.model import BaselineModel
from src.train_baseline import decode_greedy


# ---------------------------------------------------------------------------
# Optional: beam-search decoding via pyctcdecode
# ---------------------------------------------------------------------------

def _build_beam_decoder(text_transform, beam_width: int = 50):
    """Build a pyctcdecode beam decoder. Returns None if unavailable."""
    try:
        from pyctcdecode import build_ctcdecoder
        # pyctcdecode expects upper-case vocabulary; we store lower-case,
        # so we keep them lower-case and pass hotwords/lm as None (no LM).
        vocab = list(text_transform.chars) + ['']   # '' represents blank
        decoder = build_ctcdecoder(
            labels=[c.upper() if c != ' ' else ' ' for c in vocab],
            kenlm_model=None,
        )
        return decoder
    except ImportError:
        return None


def decode_beam(log_probs: torch.Tensor, blank: int,
                text_transform, beam_width: int = 50) -> list[list[int]]:
    """
    Beam-search CTC decoding via pyctcdecode.
    Falls back to greedy if pyctcdecode is unavailable.

    Args:
        log_probs:  (B, T, vocab_size)  — already log-softmax'd
        blank:      blank token index
        beam_width: number of beams
    Returns:
        list of int sequences (one per sample)
    """
    decoder = _build_beam_decoder(text_transform, beam_width)
    if decoder is None:
        print('[warning] pyctcdecode not found, falling back to greedy decoding')
        return decode_greedy(log_probs, blank)

    import numpy as np
    # pyctcdecode expects probs (not log-probs) as numpy array (T, vocab_size)
    probs = log_probs.float().exp().cpu().numpy()

    results = []
    chars = text_transform.chars
    for i in range(probs.shape[0]):
        text = decoder.decode(probs[i], beam_width=beam_width)
        # Convert decoded string back to int sequence
        text_lower = text.lower()
        seq = []
        for c in text_lower:
            if c in chars:
                seq.append(chars.index(c))
        results.append(seq)
    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: BaselineModel,
    loader: DataLoader,
    text_transform,
    blank: int,
    device,
    beam_width: int = 0,
) -> dict:
    """
    Returns a dict with keys:
        wer        : word error rate
        cer        : character error rate
        n_samples  : number of evaluated utterances
        hypotheses : list of decoded strings
        references : list of reference strings
    """
    model.eval()
    hypotheses, references = [], []

    for batch in loader:
        raw_emg   = batch['raw_emg'].to(device)
        text_int  = batch['text_int']
        t_lengths = batch['text_int_lengths']

        out       = model(raw_emg)
        log_probs = out['log_probs']   # (B, T, V)

        if beam_width > 0:
            decoded = decode_beam(log_probs, blank, text_transform, beam_width)
        else:
            decoded = decode_greedy(log_probs, blank)

        for i, pred_ints in enumerate(decoded):
            hyp = text_transform.int_to_text(pred_ints)
            ref = text_transform.int_to_text(
                text_int[i, : t_lengths[i]].tolist()
            )
            hypotheses.append(hyp)
            references.append(ref)

    wer = jiwer.wer(references, hypotheses) if references else float('inf')
    cer = jiwer.cer(references, hypotheses) if references else float('inf')

    return {
        'wer':        wer,
        'cer':        cer,
        'n_samples':  len(references),
        'hypotheses': hypotheses,
        'references': references,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(checkpoint_path: str, config_path: str, split: str,
         beam_width: int) -> None:

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg_model = cfg['model']
    cfg_data  = cfg['data']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval] device: {device}')

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    text_transform = build_text_transform()
    n_vocab        = vocab_size(text_transform)
    blank          = blank_id(text_transform)

    dataset = EMGCharDataset(
        emg_data_dir=cfg_data['emg_data_dir'],
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=EMGCharDataset.collate_fn,
    )
    print(f'[eval] {split} set: {len(dataset)} utterances')

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = BaselineModel(
        vocab_size=n_vocab,
        model_size=cfg_model['model_size'],
        num_layers=cfg_model['num_layers'],
        dropout=cfg_model['dropout'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # All three run types (baseline / uml / finetune) save model.state_dict()
    # for the EMG-only branch (BaselineModel-compatible).
    # For a UML checkpoint we need to remap key names first.
    raw_state = ckpt['model']
    run_type  = ckpt.get('run_type', 'baseline')   # may be absent on older saves

    # Detect UML checkpoint by presence of 'emg_encoder.' prefix
    if any(k.startswith('emg_encoder.') for k in raw_state):
        print('[eval] Detected UML checkpoint — remapping emg_encoder.* → encoder.*')
        remapped = {}
        for k, v in raw_state.items():
            if k.startswith('emg_encoder.'):
                remapped['encoder.' + k[len('emg_encoder.'):]] = v
            elif k.startswith('audio_encoder.'):
                pass   # discard audio encoder
            else:
                remapped[k] = v
        raw_state = remapped

    model.load_state_dict(raw_state, strict=True)
    print(f'[eval] Loaded checkpoint: {checkpoint_path}  (run_type={run_type})')

    # ---------------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------------
    decode_label = f'beam(width={beam_width})' if beam_width > 0 else 'greedy'
    print(f'[eval] Decoding: {decode_label}')

    results = run_evaluation(model, loader, text_transform, blank, device, beam_width)

    print(f'\n{"="*50}')
    print(f'Split     : {split}')
    print(f'Samples   : {results["n_samples"]}')
    print(f'WER       : {results["wer"]*100:.2f} %')
    print(f'CER       : {results["cer"]*100:.2f} %')
    print(f'Decoding  : {decode_label}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'{"="*50}\n')

    # Dump first 20 examples for inspection
    print('Sample predictions (first 20):')
    for i in range(min(20, len(results['hypotheses']))):
        print(f'  REF: {results["references"][i]}')
        print(f'  HYP: {results["hypotheses"][i]}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--config', default='configs/baseline.yaml',
                        help='YAML config with model/data sections')
    parser.add_argument('--split', default='test', choices=['dev', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--beam_width', type=int, default=0,
                        help='Beam width for CTC beam search (0 = greedy)')
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.split, args.beam_width)
