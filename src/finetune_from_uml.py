"""
Fine-tune a UML-pretrained SharedTransformer on sEMG with CTC loss.

Loads the SharedTransformer weights from a UML checkpoint, builds a fresh
BaselineModel with those weights, and fine-tunes end-to-end on EMGCharDataset.

Usage:
    python src/finetune_from_uml.py --config configs/uml.yaml \
        --uml_checkpoint /scratch/cr4206/checkpoints/uml/best.pt

Saves checkpoints to <checkpoint_dir>/finetune/.
Compatible output format with train_baseline.py — evaluate.py can load either.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb
import jiwer

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import EMGCharDataset, build_text_transform, vocab_size, blank_id
from src.model import BaselineModel, UMLModel
from src.train_baseline import decode_greedy, evaluate, build_scheduler


# ---------------------------------------------------------------------------
# Weight transfer helper
# ---------------------------------------------------------------------------

def load_transformer_from_uml(uml_ckpt_path: str, baseline_model: BaselineModel,
                               device) -> None:
    """
    Copy SharedTransformer weights from a UML checkpoint into baseline_model.

    The UML checkpoint contains a UMLModel state_dict.  Keys under
    'transformer.*' map directly to 'transformer.*' in BaselineModel.
    EMGEncoder weights are also transferred (same architecture).
    """
    ckpt = torch.load(uml_ckpt_path, map_location=device)
    uml_state = ckpt['model']

    baseline_state = baseline_model.state_dict()
    transferred = 0
    skipped = []

    for key, val in uml_state.items():
        # UMLModel keys: emg_encoder.*, transformer.*, ctc_head.*
        # BaselineModel keys: encoder.*, transformer.*, ctc_head.*
        if key.startswith('emg_encoder.'):
            mapped = 'encoder.' + key[len('emg_encoder.'):]
        elif key.startswith('audio_encoder.'):
            # audio encoder has no counterpart in baseline — skip
            skipped.append(key)
            continue
        else:
            mapped = key   # transformer.* and ctc_head.* match directly

        if mapped in baseline_state and baseline_state[mapped].shape == val.shape:
            baseline_state[mapped] = val
            transferred += 1
        else:
            skipped.append(key)

    baseline_model.load_state_dict(baseline_state)
    print(f'[transfer] {transferred} tensors transferred from UML checkpoint')
    if skipped:
        print(f'[transfer] {len(skipped)} keys skipped: {skipped[:5]}{"..." if len(skipped) > 5 else ""}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, uml_checkpoint: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg_model    = cfg['model']
    cfg_training = cfg['training']
    cfg_data     = cfg['data']
    cfg_logging  = cfg['logging']
    log_every_steps = int(cfg_training.get('log_every_steps', 200))

    torch.manual_seed(42)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = os.path.join(cfg_logging['checkpoint_dir'], 'finetune')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    text_transform = build_text_transform()
    n_vocab        = vocab_size(text_transform)
    blank          = blank_id(text_transform)

    emg_cache_dir = cfg_data['emg_cache_dir']

    train_dataset = EMGCharDataset(cache_path=emg_cache_dir, split='train')
    val_dataset   = EMGCharDataset(cache_path=emg_cache_dir, split='dev')

    batch_size = cfg_training['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=EMGCharDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=EMGCharDataset.collate_fn,
    )

    # ---------------------------------------------------------------------------
    # Model — build BaselineModel then transfer UML weights
    # ---------------------------------------------------------------------------
    model = BaselineModel(
        vocab_size=n_vocab,
        model_size=cfg_model['model_size'],
        num_layers=cfg_model['num_layers'],
        dropout=cfg_model['dropout'],
    ).to(device)

    print(f'[init] Loading UML weights from: {uml_checkpoint}')
    load_transformer_from_uml(uml_checkpoint, model, device)

    # ---------------------------------------------------------------------------
    # Optimizer & scheduler (same as baseline)
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
    wandb_init_kwargs = {
        'project': cfg_logging['wandb_project'],
        'name': 'finetune_from_uml',
        'config': {**cfg, 'uml_checkpoint': uml_checkpoint},
        'mode': 'offline',
    }
    if cfg_logging.get('wandb_entity'):
        wandb_init_kwargs['entity'] = cfg_logging['wandb_entity']
    wandb.init(**wandb_init_kwargs)
    wandb.watch(model, log_freq=200)

    # Shared x-axes so baseline / UML / finetune runs overlay cleanly in wandb.
    wandb.define_metric('epoch')
    wandb.define_metric('emg_samples_seen')
    wandb.define_metric('val/*',              step_metric='epoch')
    wandb.define_metric('train/loss_epoch',   step_metric='epoch')
    wandb.define_metric('train/loss',         step_metric='emg_samples_seen')

    # ---------------------------------------------------------------------------
    # Resume if a finetune checkpoint already exists
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
        print(f'[resume] starting from epoch {start_epoch}')

    # ---------------------------------------------------------------------------
    # Training (identical loop to train_baseline.py)
    # ---------------------------------------------------------------------------
    n_epochs = cfg_training['n_epochs']

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            raw_emg   = batch['raw_emg'].to(device)
            text_int  = batch['text_int'].to(device)
            lengths   = batch['lengths'].to(device)
            t_lengths = batch['text_int_lengths'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out  = model(
                    raw_emg,
                    return_loss=True,
                    targets=text_int,
                    input_lengths=lengths,
                    target_lengths=t_lengths,
                )
            loss = out['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            warmup_sched.step()

            global_step += 1
            epoch_loss  += loss.item()
            n_batches   += 1

            if global_step % log_every_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f'[train] epoch={epoch+1} batch={n_batches} '
                    f'step={global_step} loss={loss.item():.4f} lr={lr:.2e}',
                    flush=True,
                )
                wandb.log({
                    'train/loss':        loss.item(),
                    'emg_samples_seen':  global_step * batch_size,
                    'step':              global_step,
                })

        if epoch + 1 in milestones:
            lr_decay[0] *= gamma
            print(f'[lr decay] epoch {epoch+1}: lr → {optimizer.param_groups[0]["lr"] * gamma:.2e}')

        avg_loss = epoch_loss / max(n_batches, 1)
        val_wer  = evaluate(model, val_loader, text_transform, blank, device)

        print(f'epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  val_wer={val_wer:.4f}')
        wandb.log({
            'epoch':            epoch + 1,
            'train/loss_epoch': avg_loss,
            'val/wer':          val_wer,
        })

        ckpt_data = {
            'epoch':          epoch,
            'global_step':    global_step,
            'model':          model.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'val_wer':        val_wer,
            'config':         cfg,
            'uml_checkpoint': uml_checkpoint,
            # Tag allows evaluate.py to identify origin
            'run_type':       'finetune',
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
    print('Fine-tuning complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/uml.yaml',
                        help='YAML config (model/training/data sections must match UML run)')
    parser.add_argument('--uml_checkpoint', required=True,
                        help='Path to UML checkpoint (e.g. $SCRATCH/checkpoints/uml/best.pt)')
    args = parser.parse_args()
    main(args.config, args.uml_checkpoint)
