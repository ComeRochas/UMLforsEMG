"""
Train the baseline model: EMGEncoder → SharedTransformer → CTCHead.

Usage:
    python src/train_baseline.py --config configs/baseline.yaml

Logs train loss and val WER to Weights & Biases.
Saves checkpoints to <checkpoint_dir>/baseline/.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb
import jiwer

# ---------------------------------------------------------------------------
# Project root on sys.path (for read_emg / data_utils / transformer imports)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import EMGCharDataset, build_text_transform, vocab_size, blank_id
from src.model import BaselineModel


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------

def _warmup_factor(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    return 1.0


def build_scheduler(optimizer, cfg_training):
    warmup_steps = cfg_training['warmup_steps']
    milestones   = cfg_training.get('lr_milestones', [125, 150, 175])
    gamma        = cfg_training.get('lr_gamma', 0.5)

    # Mutable so the epoch-decay code can update it while the lambda still
    # references it.  Without this, LambdaLR.step() always recomputes LR as
    # base_lr * warmup_factor, silently overwriting any pg['lr'] *= gamma.
    lr_decay = [1.0]

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_factor(step, warmup_steps) * lr_decay[0],
    )
    return warmup_sched, milestones, gamma, lr_decay


# ---------------------------------------------------------------------------
# WER evaluation
# ---------------------------------------------------------------------------

def decode_greedy(log_probs: torch.Tensor, blank: int) -> list[list[int]]:
    """
    Greedy CTC decode.

    Args:
        log_probs: (B, T, vocab_size)
        blank:     blank token index
    Returns:
        list of lists of int (one per sample)
    """
    preds = log_probs.argmax(dim=-1)  # (B, T)
    results = []
    for seq in preds:
        decoded = []
        prev = blank
        for tok in seq.tolist():
            if tok != blank and tok != prev:
                decoded.append(tok)
            prev = tok
        results.append(decoded)
    return results


@torch.no_grad()
def evaluate(model: BaselineModel, loader: DataLoader,
             text_transform, blank: int, device) -> float:
    model.eval()
    all_hyps, all_refs = [], []
    for batch in loader:
        raw_emg  = batch['raw_emg'].to(device)
        text_int = batch['text_int']
        t_lengths = batch['text_int_lengths']

        out = model(raw_emg)
        log_probs = out['log_probs']                # (B, T, V)

        decoded = decode_greedy(log_probs, blank)
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
# Main training loop
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg_model    = cfg['model']
    cfg_training = cfg['training']
    cfg_data     = cfg['data']
    cfg_logging  = cfg['logging']
    log_every_steps = int(cfg_training.get('log_every_steps', 20))

    # Reproducibility
    torch.manual_seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[startup] device={device}', flush=True)

    # Checkpoint dir
    ckpt_dir = os.path.join(cfg_logging['checkpoint_dir'], 'baseline')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'[startup] checkpoint_dir={ckpt_dir}', flush=True)

    # ---------------------------------------------------------------------------
    # Datasets & dataloaders
    # ---------------------------------------------------------------------------
    text_transform = build_text_transform()
    n_vocab        = vocab_size(text_transform)
    blank          = blank_id(text_transform)

    emg_data_dir = cfg_data['emg_data_dir']

    train_dataset = EMGCharDataset(emg_data_dir=emg_data_dir, split='train')
    val_dataset   = EMGCharDataset(emg_data_dir=emg_data_dir, split='dev')
    print(
        f'[data] train_samples={len(train_dataset)} val_samples={len(val_dataset)} '
        f'batch_size={cfg_training["batch_size"]}',
        flush=True,
    )

    batch_size = cfg_training['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=EMGCharDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=EMGCharDataset.collate_fn,
    )

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = BaselineModel(
        vocab_size=n_vocab,
        model_size=cfg_model['model_size'],
        num_layers=cfg_model['num_layers'],
        dropout=cfg_model['dropout'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] parameters={n_params:,}', flush=True)

    # ---------------------------------------------------------------------------
    # Optimizer & scheduler
    # ---------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_training['learning_rate'],
        weight_decay=cfg_training.get('l2', 0.0),
    )

    warmup_sched, milestones, gamma, lr_decay = build_scheduler(optimizer, cfg_training)
    global_step = 0

    # ---------------------------------------------------------------------------
    # W&B
    # ---------------------------------------------------------------------------
    wandb.init(
        project=cfg_logging['wandb_project'],
        name='baseline',
        config=cfg,
    )
    wandb.watch(model, log_freq=200)
    print(f'[wandb] initialized project={cfg_logging["wandb_project"]}', flush=True)

    # ---------------------------------------------------------------------------
    # Resume from checkpoint if present
    # ---------------------------------------------------------------------------
    start_epoch = 0
    latest_ckpt = os.path.join(ckpt_dir, 'latest.pt')
    if os.path.isfile(latest_ckpt):
        print(f'[resume] loading {latest_ckpt}')
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch  = ckpt['epoch'] + 1
        global_step  = ckpt.get('global_step', 0)
        print(f'[resume] starting from epoch {start_epoch}')

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    n_epochs = cfg_training['n_epochs']

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        epoch_start = time.time()
        print(f'[epoch {epoch+1}/{n_epochs}] started', flush=True)

        for batch in train_loader:
            raw_emg      = batch['raw_emg'].to(device)          # (B, T_raw, 8)
            text_int     = batch['text_int'].to(device)         # (B, L)
            lengths      = batch['lengths'].to(device)           # (B,) EMG frames
            t_lengths    = batch['text_int_lengths'].to(device)  # (B,)

            out  = model(
                raw_emg,
                return_loss=True,
                targets=text_int,
                input_lengths=lengths,
                target_lengths=t_lengths,
            )
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            warmup_sched.step()

            global_step += 1
            epoch_loss  += loss.item()
            n_batches   += 1

            if n_batches % log_every_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start
                print(
                    f'[train] epoch={epoch+1} batch={n_batches} '
                    f'step={global_step} loss={loss.item():.4f} '
                    f'lr={lr:.2e} elapsed={elapsed:.1f}s',
                    flush=True,
                )

            if global_step % 50 == 0:
                wandb.log({'train/loss': loss.item(), 'step': global_step})

        # Epoch-level LR decay — update the closure variable so LambdaLR
        # picks it up on the next warmup_sched.step() call.
        if epoch + 1 in milestones:
            lr_decay[0] *= gamma
            print(f'[lr decay] epoch {epoch+1}: lr → {optimizer.param_groups[0]["lr"] * gamma:.2e}')

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation WER
        val_wer = evaluate(model, val_loader, text_transform, blank, device)

        print(f'epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  val_wer={val_wer:.4f}')
        wandb.log({
            'epoch':        epoch + 1,
            'train/loss_epoch': avg_loss,
            'val/wer':      val_wer,
        })

        # Save checkpoint
        ckpt_data = {
            'epoch':       epoch,
            'global_step': global_step,
            'model':       model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'val_wer':     val_wer,
            'config':      cfg,
        }
        torch.save(ckpt_data, latest_ckpt)

        # Also save best
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
    parser.add_argument('--config', default='configs/baseline.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)
