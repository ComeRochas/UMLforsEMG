"""
Train the UML model: dual-branch (EMG + audio) sharing a single Transformer.

Loss:  loss = loss_emg + lambda_uml * loss_audio

Alternates batches (one EMG, one audio) and accumulates gradients over 2 steps
before calling optimizer.step(), so both modalities contribute to every update.
AudioEncoder is always frozen.

Usage:
    python src/train_uml.py --config configs/uml.yaml

Logs train/emg_loss, train/audio_loss, val/wer to Weights & Biases.
Saves checkpoints to <checkpoint_dir>/uml/.
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb
import jiwer

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import (
    EMGCharDataset,
    LibriSpeechCharDataset,
    build_text_transform,
    vocab_size,
    blank_id,
)
from src.model import UMLModel
from src.train_baseline import decode_greedy   # shared helper


# ---------------------------------------------------------------------------
# Scheduler helpers (same warmup + epoch-based milestone decay)
# ---------------------------------------------------------------------------

def _warmup_factor(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    return 1.0


def build_scheduler(optimizer, cfg_training):
    warmup_steps = cfg_training['warmup_steps']
    milestones   = cfg_training.get('lr_milestones', [125, 150, 175])
    gamma        = cfg_training.get('lr_gamma', 0.5)

    lr_decay = [1.0]

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_factor(step, warmup_steps) * lr_decay[0],
    )
    return warmup_sched, milestones, gamma, lr_decay


# ---------------------------------------------------------------------------
# WER evaluation (EMG branch only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: UMLModel, loader: DataLoader,
             text_transform, blank: int, device) -> float:
    model.eval()
    all_hyps, all_refs = [], []
    for batch in loader:
        raw_emg   = batch['raw_emg'].to(device)
        text_int  = batch['text_int']
        t_lengths = batch['text_int_lengths']

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model.forward_emg(raw_emg)
        decoded = decode_greedy(out['log_probs'], blank)
        for i, pred_ints in enumerate(decoded):
            hyp = text_transform.int_to_text(pred_ints)
            ref = text_transform.int_to_text(
                text_int[i, : t_lengths[i]].tolist()
            )
            all_hyps.append(hyp)
            all_refs.append(ref)

    if not all_refs:
        return float('inf')
    wer = jiwer.wer(all_refs, all_hyps)
    model.train()
    return wer


# ---------------------------------------------------------------------------
# Infinite DataLoader iterator
# ---------------------------------------------------------------------------

def infinite_loader(loader: DataLoader):
    """Cycle through a DataLoader indefinitely."""
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg_model    = cfg['model']
    cfg_training = cfg['training']
    cfg_data     = cfg['data']
    cfg_logging  = cfg['logging']
    cfg_uml      = cfg['uml']
    log_every_steps = int(cfg_training.get('log_every_steps', 200))

    torch.manual_seed(42)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Checkpoint dir
    ckpt_dir = os.path.join(cfg_logging['checkpoint_dir'], 'uml')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Datasets & dataloaders
    # ---------------------------------------------------------------------------
    text_transform = build_text_transform()
    n_vocab        = vocab_size(text_transform)
    blank          = blank_id(text_transform)

    emg_cache_dir        = cfg_data['emg_cache_dir']
    librispeech_cache_dir = cfg_data['librispeech_cache_dir']
    libri_split          = cfg_data.get('librispeech_split', 'train-clean-100')

    # EMG datasets (precomputed cache)
    emg_train = EMGCharDataset(cache_path=emg_cache_dir, split='train')
    emg_val   = EMGCharDataset(cache_path=emg_cache_dir, split='dev')

    # LibriSpeech dataset (precomputed cache)
    libri_train = LibriSpeechCharDataset(
        cache_path=librispeech_cache_dir,
        split=libri_split,
    )

    batch_size = cfg_training['batch_size']

    # All datasets are already in memory (precomputed).  num_workers=2 is
    # enough to hide pad/stack behind GPU compute.
    emg_loader = DataLoader(
        emg_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=EMGCharDataset.collate_fn,
        drop_last=True,
    )
    audio_loader = DataLoader(
        libri_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=LibriSpeechCharDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        emg_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=EMGCharDataset.collate_fn,
    )

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = UMLModel(
        vocab_size=n_vocab,
        model_size=cfg_model['model_size'],
        num_layers=cfg_model['num_layers'],
        dropout=cfg_model['dropout'],
    ).to(device)

    # Mutable audio-loss state — the scalar is read per-batch below.
    # If `lambda_patience > 0`, the scalar is halved (by `lambda_decay`) after
    # `lambda_patience` epochs without a val_wer improvement of at least
    # `lambda_tol`, down to `lambda_min`.  Off by default.
    lambda_state = {
        'value':         float(cfg_uml['lambda_uml']),
        'best_wer':      float('inf'),
        'stale_epochs':  0,
        'patience':      int(cfg_uml.get('lambda_patience', 0)),
        'decay':         float(cfg_uml.get('lambda_decay', 0.5)),
        'min':           float(cfg_uml.get('lambda_min', 0.0)),
        'tol':           float(cfg_uml.get('lambda_tol', 1e-3)),
    }

    # ---------------------------------------------------------------------------
    # Optimizer  — AudioEncoder parameters are frozen; exclude them explicitly
    # ---------------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg_training['learning_rate'],
        weight_decay=cfg_training.get('l2', 0.0),
    )

    warmup_sched, milestones, gamma, lr_decay = build_scheduler(optimizer, cfg_training)
    global_step = 0

    # ---------------------------------------------------------------------------
    # W&B
    # ---------------------------------------------------------------------------
    wandb_init_kwargs = {
        'project': cfg_logging['wandb_project'],
        'name': 'uml',
        'config': cfg,
        'mode': 'offline',
    }
    if cfg_logging.get('wandb_entity'):
        wandb_init_kwargs['entity'] = cfg_logging['wandb_entity']
    wandb.init(**wandb_init_kwargs)
    wandb.watch(model, log_freq=200)

    # Shared x-axes so baseline / UML / finetune runs overlay cleanly in wandb.
    wandb.define_metric('epoch')
    wandb.define_metric('emg_samples_seen')
    wandb.define_metric('val/*',                      step_metric='epoch')
    wandb.define_metric('train/emg_loss_epoch',       step_metric='epoch')
    wandb.define_metric('train/audio_loss_epoch',     step_metric='epoch')
    wandb.define_metric('uml/lambda',                 step_metric='epoch')
    wandb.define_metric('train/emg_loss',             step_metric='emg_samples_seen')
    wandb.define_metric('train/audio_loss',           step_metric='emg_samples_seen')
    wandb.define_metric('train/total_loss',           step_metric='emg_samples_seen')

    # ---------------------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------------------
    start_epoch = 0
    latest_ckpt = os.path.join(ckpt_dir, 'latest.pt')
    if os.path.isfile(latest_ckpt):
        print(f'[resume] loading {latest_ckpt}')
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        if 'lambda_state' in ckpt:
            # Restore tunable keys from ckpt; keep config-driven knobs
            # (patience / decay / min / tol) as whatever the current YAML says,
            # so tweaking the config then resuming takes effect.
            saved = ckpt['lambda_state']
            for k in ('value', 'best_wer', 'stale_epochs'):
                if k in saved:
                    lambda_state[k] = saved[k]
        print(f'[resume] starting from epoch {start_epoch} '
              f'(lambda_uml={lambda_state["value"]:.3f})')

    # ---------------------------------------------------------------------------
    # Training
    # The inner loop interleaves one EMG batch and one audio batch,
    # accumulating gradients across both before stepping the optimizer.
    # This ensures every parameter update sees both modalities.
    # ---------------------------------------------------------------------------
    n_epochs = cfg_training['n_epochs']

    # We define one "step" as the pair (EMG batch, audio batch).
    # Steps per epoch = len(emg_loader).
    audio_iter = infinite_loader(audio_loader)

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_emg_loss   = 0.0
        epoch_audio_loss = 0.0
        n_batches        = 0

        optimizer.zero_grad()

        for step_in_epoch, emg_batch in enumerate(emg_loader):
            # ---- Step 1: EMG batch ----------------------------------------
            raw_emg   = emg_batch['raw_emg'].to(device)
            text_int  = emg_batch['text_int'].to(device)
            lengths   = emg_batch['lengths'].to(device)
            t_lengths = emg_batch['text_int_lengths'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                emg_out  = model.forward_emg(
                    raw_emg,
                    return_loss=True,
                    targets=text_int,
                    input_lengths=lengths,
                    target_lengths=t_lengths,
                )
            loss_emg = emg_out['loss']

            # Accumulate (divide by 2 for gradient accumulation over 2 sub-steps)
            (loss_emg / 2).backward()

            # ---- Step 2: Audio batch (AudioEncoder frozen internally) -------
            audio_batch   = next(audio_iter)
            waveform      = audio_batch['audio_features'].to(device)
            a_text_int    = audio_batch['text_int'].to(device)
            a_t_lengths   = audio_batch['text_int_lengths'].to(device)
            audio_lengths = audio_batch['audio_lengths'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                audio_out = model.forward_audio(
                    waveform, a_text_int, a_t_lengths, audio_lengths=audio_lengths
                )
            loss_audio  = audio_out['loss']

            combined = lambda_state['value'] * loss_audio
            (combined / 2).backward()

            # ---- Optimizer step after both sub-steps -----------------------
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            warmup_sched.step()
            optimizer.zero_grad()

            global_step      += 1
            epoch_emg_loss   += loss_emg.item()
            epoch_audio_loss += loss_audio.item()
            n_batches        += 1

            if global_step % log_every_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f'[train] epoch={epoch+1} batch={n_batches} '
                    f'step={global_step} emg_loss={loss_emg.item():.4f} '
                    f'audio_loss={loss_audio.item():.4f} lr={lr:.2e}',
                    flush=True,
                )
                wandb.log({
                    'train/emg_loss':   loss_emg.item(),
                    'train/audio_loss': loss_audio.item(),
                    'train/total_loss': loss_emg.item() + lambda_state['value'] * loss_audio.item(),
                    'emg_samples_seen': global_step * batch_size,
                    'step':             global_step,
                })

        # Epoch-level LR decay
        if epoch + 1 in milestones:
            lr_decay[0] *= gamma
            print(f'[lr decay] epoch {epoch+1}: lr → {optimizer.param_groups[0]["lr"] * gamma:.2e}')

        avg_emg   = epoch_emg_loss   / max(n_batches, 1)
        avg_audio = epoch_audio_loss / max(n_batches, 1)

        val_wer = evaluate(model, val_loader, text_transform, blank, device)

        # Patience-based lambda decay (robust to noisy WER via lambda_tol).
        # Disabled when patience == 0.
        if lambda_state['patience'] > 0:
            if val_wer < lambda_state['best_wer'] - lambda_state['tol']:
                lambda_state['best_wer']     = val_wer
                lambda_state['stale_epochs'] = 0
            else:
                lambda_state['stale_epochs'] += 1
                if lambda_state['stale_epochs'] >= lambda_state['patience']:
                    new_val = max(
                        lambda_state['min'],
                        lambda_state['value'] * lambda_state['decay'],
                    )
                    if new_val < lambda_state['value']:
                        print(
                            f'[lambda decay] val_wer stale for '
                            f'{lambda_state["stale_epochs"]} epochs → '
                            f'lambda_uml {lambda_state["value"]:.3f} → {new_val:.3f}',
                            flush=True,
                        )
                        lambda_state['value'] = new_val
                    lambda_state['stale_epochs'] = 0

        print(
            f'epoch {epoch+1}/{n_epochs}  '
            f'emg_loss={avg_emg:.4f}  '
            f'audio_loss={avg_audio:.4f}  '
            f'val_wer={val_wer:.4f}  '
            f'lambda={lambda_state["value"]:.3f}'
        )
        wandb.log({
            'epoch':                  epoch + 1,
            'train/emg_loss_epoch':   avg_emg,
            'train/audio_loss_epoch': avg_audio,
            'val/wer':                val_wer,
            'uml/lambda':             lambda_state['value'],
        })

        # Save checkpoint
        ckpt_data = {
            'epoch':        epoch,
            'global_step':  global_step,
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'val_wer':      val_wer,
            'config':       cfg,
            'lambda_state': lambda_state,
        }
        torch.save(ckpt_data, latest_ckpt)

        best_ckpt = os.path.join(ckpt_dir, 'best.pt')
        if not os.path.isfile(best_ckpt):
            torch.save(ckpt_data, best_ckpt)
        else:
            prev = torch.load(best_ckpt, map_location='cpu')
            if val_wer < prev.get('val_wer', float('inf')):
                torch.save(ckpt_data, best_ckpt)
                print(f'  [best] saved (val_wer={val_wer:.4f})')

    wandb.finish()
    print('Training complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/uml.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)
