# silent_speech_uml

Testing whether **UML** (unified multimodal learning — a single Transformer
shared across EMG and audio branches) improves sEMG encoder training.

Three training entry-points, all CTC with a shared model architecture
([src/model.py](src/model.py)):

| Script | What it does | Checkpoint dir |
|---|---|---|
| [src/train_baseline.py](src/train_baseline.py) | `EMGEncoder → SharedTransformer → CTCHead`, trained end-to-end on sEMG only | `$SCRATCH/checkpoints/baseline/` |
| [src/train_uml.py](src/train_uml.py) | Dual-branch UML: EMG + LibriSpeech audio share the same Transformer; AudioEncoder frozen (wav2vec2-base); aux loss `loss = loss_emg + λ·loss_audio` | `$SCRATCH/checkpoints/uml/` |
| [src/finetune_from_uml.py](src/finetune_from_uml.py) | Transfer SharedTransformer + EMGEncoder weights from a UML checkpoint into a fresh BaselineModel, then fine-tune on sEMG | `$SCRATCH/checkpoints/finetune/` |

Evaluation on `dev`/`test`: [src/evaluate.py](src/evaluate.py).

## Repo layout

```
src/
  precompute_emg.py        ← run ONCE on CPU to produce train.pt / dev.pt / test.pt
  precompute_audio.py      ← run ONCE on CPU to produce train-clean-100.pt
  data.py                  ← EMGCharDataset + LibriSpeechCharDataset (both cache readers)
  model.py                 ← EMGEncoder, AudioEncoder, SharedTransformer, CTCHead, BaselineModel, UMLModel
  train_baseline.py
  train_uml.py
  finetune_from_uml.py
  evaluate.py
configs/
  baseline.yaml
  uml.yaml
slurm/
  train_baseline.slurm
  train_uml.slurm
  finetune_from_uml.slurm
scripts/
  download_data.sh         ← fetches Gaddy sEMG + LibriSpeech
data_utils.py              ← TextTransform (char vocab)
testset_largedev.json      ← dev/test split definition (used by precompute_emg)
environment.yml
```

## Why there's a precompute step

Gaddy's live EMG `__getitem__` does ~10× more CPU work per sample than the
model consumes (`get_emg_features`, MFCC/librosa audio, TextGrid phonemes,
mfcc/emg normalizers, silent→voiced parallel lookup, per-sample pinning).
LibriSpeech's live `__getitem__` spends its time in FLAC decoding + per-sample
Wav2Vec2 normalization. Both were starving the GPU.

[src/precompute_emg.py](src/precompute_emg.py) runs the EMG signal chain **once**
on CPU and materializes only what the model needs:

* notch harmonics (60 Hz × 1..7) + 2 Hz highpass drift-removal
* subsample 1000 Hz → 689.06 Hz
* normalize `raw_emg / 20`, then `50·tanh(raw_emg/50)`
* cap at 6400 frames (`limit_length=True`)
* char-level `text_int`

[src/precompute_audio.py](src/precompute_audio.py) does the same for
LibriSpeech:

* FLAC decode → fp32 waveform (+ resample to 16 kHz if needed — a no-op for
  LibriSpeech)
* per-sample zero-mean / unit-variance normalization (same formula as
  `Wav2Vec2FeatureExtractor`, so we can drop `transformers` from the
  runtime path)
* char-level `text_int`

Result at `$SCRATCH/data/emg_cache/{split}.pt` and
`$SCRATCH/data/libri_cache/{split}.pt` — each a dict of fp16 tensors.  EMG:
~1 GB total. LibriSpeech `train-clean-100`: ~11 GB. At train time,
`__getitem__` is a list lookup on both branches — CPU cost per batch
collapses and the GPU is no longer starved.

## Data pipeline quick reference

| Step | Where | Notes |
|---|---|---|
| Download Gaddy sEMG + LibriSpeech | [scripts/download_data.sh](scripts/download_data.sh) | ~4 GB + ~6 GB, extracted to `$SCRATCH/data/` |
| Precompute EMG cache | [src/precompute_emg.py](src/precompute_emg.py) | **CPU-only, one-shot**, ~10–20 min on 8 cores |
| Precompute audio cache | [src/precompute_audio.py](src/precompute_audio.py) | **CPU-only, one-shot** (UML only), ~15–30 min on 8 cores |
| Train | `src/train_*.py` | reads `$SCRATCH/data/{emg_cache,libri_cache}/*.pt` |

Baseline and finetune only read the EMG cache; `precompute_audio.py` is only
needed before running `train_uml.py`.

## How to run

### 0. One-off setup (done once)

```bash
export SCRATCH=/scratch/cr4206
# Conda env at $SCRATCH to avoid exhausting $HOME file quota
module load anaconda3/2025.06
conda env create --prefix $SCRATCH/envs/silent_speech_uml -f environment.yml

# Fetch raw data (Gaddy sEMG + LibriSpeech)
bash scripts/download_data.sh
```

### 1. Precompute EMG tensors (CPU only — no GPU needed)

Run this from any machine with the conda env + access to the raw data.
A compute node is fine but not required; the login node works if you're
polite about CPU use.

```bash
conda activate $SCRATCH/envs/silent_speech_uml
cd $HOME/silent_speech_uml

python -u src/precompute_emg.py \
    --emg_data_dir $SCRATCH/data/emg_data/emg_data \
    --out_dir      $SCRATCH/data/emg_cache \
    --num_workers  8
```

Monitor the output: the script prints `[<split>] i/N  rate=X samp/s  eta=Ys`
about 40× per split. Expected total: **~10–20 min** for all three splits on
8 CPU cores. The resulting files:

```
$SCRATCH/data/emg_cache/train.pt   # ~1 GB
$SCRATCH/data/emg_cache/dev.pt
$SCRATCH/data/emg_cache/test.pt
```

### 1b. Precompute LibriSpeech tensors (CPU only — only needed before `train_uml`)

Skip this step if you're only running baseline + finetune.

```bash
python -u src/precompute_audio.py \
    --librispeech_dir $SCRATCH/data/librispeech \
    --out_dir         $SCRATCH/data/libri_cache \
    --splits          train-clean-100 \
    --num_workers     8
```

Writes `$SCRATCH/data/libri_cache/train-clean-100.pt` (~11 GB).  Expected
runtime: **~15–30 min** on 8 CPU cores.

### 2. Train the baseline

```bash
sbatch slurm/train_baseline.slurm
```

Or interactively:

```bash
conda activate $SCRATCH/envs/silent_speech_uml
python -u src/train_baseline.py --config configs/baseline.yaml
```

The slurm script pre-flight-checks the cache and aborts with a clear error
if `train.pt` / `dev.pt` are missing.

### 3. Train UML

```bash
sbatch slurm/train_uml.slurm
```

Reads both the EMG cache (`$SCRATCH/data/emg_cache/`) and the LibriSpeech
cache (`$SCRATCH/data/libri_cache/train-clean-100.pt`).  The slurm script
pre-flight-checks both and aborts with a clear error if either is missing.

### 4. Fine-tune from a UML checkpoint

```bash
sbatch slurm/finetune_from_uml.slurm
```

By default loads `$SCRATCH/checkpoints/uml/best.pt`. Override via:

```bash
python -u src/finetune_from_uml.py \
    --config configs/uml.yaml \
    --uml_checkpoint /path/to/uml/best.pt
```

### 5. Evaluate any checkpoint

```bash
python -u src/evaluate.py \
    --checkpoint $SCRATCH/checkpoints/baseline/best.pt \
    --config     configs/baseline.yaml \
    --split      test
```

Works for baseline, UML, and finetune checkpoints (the script detects UML
checkpoints by key prefix and remaps `emg_encoder.*` → `encoder.*`).

## Configs

Model is small and uses native PyTorch attention
(`nn.TransformerEncoderLayer(batch_first=True, norm_first=True)` dispatches
to `scaled_dot_product_attention` → FlashAttention on Ampere/Hopper):

```yaml
model:
  model_size: 256
  num_layers: 4
  dropout:    0.1
```

Change `model_size` / `num_layers` / `batch_size` / `n_epochs` in
[configs/baseline.yaml](configs/baseline.yaml) or
[configs/uml.yaml](configs/uml.yaml). UML adds a `uml.lambda_uml`
(default 0.5) scaling the auxiliary audio loss.

## Where to change common settings

| Setting | File |
|---|---|
| `n_epochs`, `batch_size`, `learning_rate`, `log_every_steps`, `lr_milestones` | [configs/baseline.yaml](configs/baseline.yaml) / [configs/uml.yaml](configs/uml.yaml) |
| GPU type / partition (`#SBATCH --partition=`, `--gres=gpu:...`) | [slurm/train_baseline.slurm](slurm/train_baseline.slurm), [slurm/train_uml.slurm](slurm/train_uml.slurm), [slurm/finetune_from_uml.slurm](slurm/finetune_from_uml.slurm) |
| Cache locations (`emg_cache_dir`, `librispeech_cache_dir`) | the same YAML configs |
| UML audio loss weight (`lambda_uml`) | [configs/uml.yaml](configs/uml.yaml) |

A submitted slurm job freezes its config at submission time — change the
YAML and re-submit to pick up new values.

## When to re-precompute

Only if any of these change:

* EMG: filter params in `src/precompute_emg.py` (notch Q, highpass cutoff,
  subsample rate), or the `LIMIT_LENGTH_MAX_RAW` cap
* Audio: the normalization formula in `src/precompute_audio.py`
* the `TextTransform` vocabulary (affects both caches)
* `testset_largedev.json` (dev/test definition — EMG only)

Architecture / model size / batch size / LR changes do **not** require
re-precomputing.
