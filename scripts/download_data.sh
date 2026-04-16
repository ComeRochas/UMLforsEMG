#!/usr/bin/env bash
# Download and extract sEMG (Gaddy) and LibriSpeech datasets.
# Resumable: already-present files and directories are skipped.
# Usage:  bash scripts/download_data.sh
set -euo pipefail

EMG_DIR="${SCRATCH:-/scratch/cr4206}/data/emg_data"
LIBRI_DIR="${SCRATCH:-/scratch/cr4206}/data/librispeech"

mkdir -p "$EMG_DIR" "$LIBRI_DIR"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
download_if_missing() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "[skip] $(basename "$dest") already exists"
    else
        echo "[download] $(basename "$dest")"
        wget --progress=bar:force -O "$dest" "$url"
    fi
}

# ---------------------------------------------------------------------------
# Gaddy sEMG dataset (Zenodo 4064408 → redirects to 4064409)
# Only one file in the record: emg_data.tar.gz (~3.7 GB).
# normalizers.pkl and testset JSON files are bundled inside the archive.
# ---------------------------------------------------------------------------
echo "=== Gaddy sEMG dataset ==="
EMG_ARCHIVE_URL="https://zenodo.org/api/records/4064409/files/emg_data.tar.gz/content"
EMG_ARCHIVE="$EMG_DIR/emg_data.tar.gz"

download_if_missing "${EMG_ARCHIVE_URL}" "${EMG_ARCHIVE}"

# Extract the main archive if not already extracted
EXTRACT_MARKER="$EMG_DIR/.extracted"
if [[ -f "$EXTRACT_MARKER" ]]; then
    echo "[skip] sEMG archive already extracted"
else
    echo "[extract] emg_data.tar.gz → $EMG_DIR"
    tar -xzf "${EMG_ARCHIVE}" -C "$EMG_DIR"
    touch "$EXTRACT_MARKER"
    echo "[done] sEMG extraction complete"
fi

# ---------------------------------------------------------------------------
# LibriSpeech (train-clean-100 and dev-clean)
# ---------------------------------------------------------------------------
echo ""
echo "=== LibriSpeech ==="
LIBRI_BASE="https://www.openslr.org/resources/12"

LIBRI_ARCHIVES=(
    "train-clean-100.tar.gz"
    "dev-clean.tar.gz"
)

for fname in "${LIBRI_ARCHIVES[@]}"; do
    dest="$LIBRI_DIR/$fname"
    download_if_missing "${LIBRI_BASE}/${fname}" "$dest"
done

for fname in "${LIBRI_ARCHIVES[@]}"; do
    split="${fname%.tar.gz}"
    marker="$LIBRI_DIR/.extracted_${split}"
    if [[ -f "$marker" ]]; then
        echo "[skip] $split already extracted"
    else
        echo "[extract] $fname → $LIBRI_DIR"
        tar -xzf "$LIBRI_DIR/$fname" -C "$LIBRI_DIR"
        touch "$marker"
        echo "[done] $split extraction complete"
    fi
done

echo ""
echo "All data ready."
echo "  sEMG  → $EMG_DIR"
echo "  LibriSpeech → $LIBRI_DIR"
