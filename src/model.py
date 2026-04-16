"""
Model components for silent-speech UML.

Building blocks
---------------
ResBlock           — 1-D residual conv block (from Gaddy's architecture.py)
EMGEncoder         — 3 × ResBlock(stride=2) CNN for raw 8-ch EMG
AudioEncoder       — frozen wav2vec2-base + trainable projection
SharedTransformer  — N-layer PyTorch Transformer encoder w/ sinusoidal pos enc
                     (uses scaled_dot_product_attention → FlashAttention on H100)
CTCHead            — linear + log-softmax; helpers to compute CTC loss

Composite models
----------------
BaselineModel      — EMGEncoder → SharedTransformer → CTCHead
UMLModel           — dual-branch: EMG + audio through the SAME Transformer
"""
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# transformer.py (custom attention) is no longer used — SharedTransformer
# now wraps nn.TransformerEncoderLayer directly for FlashAttention-2 dispatch.

# ---------------------------------------------------------------------------
# ResBlock (verbatim from Gaddy's architecture.py, minus absl.flags)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, num_ins: int, num_outs: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm      = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value
        return F.relu(x + res)


# ---------------------------------------------------------------------------
# EMGEncoder
# ---------------------------------------------------------------------------

class EMGEncoder(nn.Module):
    """
    3 × ResBlock (stride=2) CNN that maps raw 8-channel EMG to model-dim
    feature frames.

    Input:  (B, T_raw, 8)
    Output: (B, T_raw // 8, model_size)
    """

    def __init__(self, model_size: int = 768):
        super().__init__()
        self.model_size = model_size

        self.conv_blocks = nn.Sequential(
            ResBlock(8,          model_size, stride=2),
            ResBlock(model_size, model_size, stride=2),
            ResBlock(model_size, model_size, stride=2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (B, T_raw, 8)
        Returns:
            (B, T_raw // 8, model_size)
        """
        # Training-time shift augmentation from Gaddy
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw = x_raw.clone()
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0.0

        x = x_raw.transpose(1, 2)    # (B, 8, T_raw)
        x = self.conv_blocks(x)       # (B, model_size, T_raw // 8)
        x = x.transpose(1, 2)         # (B, T_raw // 8, model_size)
        x = self.w_raw_in(x)          # (B, T_raw // 8, model_size)
        return x


# ---------------------------------------------------------------------------
# AudioEncoder  (frozen wav2vec2-base + trainable projection)
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    Encodes raw 16-kHz waveforms using facebook/wav2vec2-base (ALWAYS FROZEN).
    A trainable nn.Linear projects wav2vec2's 768-dim output to model_size.

    Input:  (B, T_audio)  — normalized waveform (from Wav2Vec2Processor)
    Output: (B, T', model_size)
    """

    WAV2VEC2_MODEL = 'facebook/wav2vec2-base'

    def __init__(self, model_size: int = 768):
        super().__init__()
        self.model_size = model_size

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(self.WAV2VEC2_MODEL)
        # Freeze all wav2vec2 parameters — never accumulate gradients
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        wav2vec2_dim = self.wav2vec2.config.hidden_size   # 768 for wav2vec2-base
        self.projection = nn.Linear(wav2vec2_dim, model_size)

    def forward(
        self,
        waveform: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform:      (B, T_audio) normalized raw waveform (zero-padded)
            audio_lengths: (B,) real sample lengths in waveform frames.
                           Used to build the attention_mask passed to wav2vec2
                           so padded frames are ignored.  If None, all frames
                           are treated as valid (single-sample inference).
        Returns:
            features:    (B, T', model_size)
            out_lengths: (B,) number of valid frames in T' dimension
        """
        # Build wav2vec2 attention mask: 1 = real, 0 = padding
        if audio_lengths is not None:
            B, T = waveform.shape
            attention_mask = (
                torch.arange(T, device=waveform.device).unsqueeze(0)
                < audio_lengths.unsqueeze(1)
            ).long()  # (B, T)
        else:
            attention_mask = None

        with torch.no_grad():
            outputs = self.wav2vec2(
                input_values=waveform,
                attention_mask=attention_mask,
            )
        # last_hidden_state: (B, T', 768)
        features = outputs.last_hidden_state

        # Compute valid output-frame counts for CTC loss
        if audio_lengths is not None:
            # wav2vec2-base downsamples by a factor of ~320 (50 Hz output for 16 kHz)
            # Use the model's utility if available, else approximate
            if hasattr(self.wav2vec2, '_get_feat_extract_output_lengths'):
                out_lengths = self.wav2vec2._get_feat_extract_output_lengths(audio_lengths)
            else:
                out_lengths = (audio_lengths - 400) // 320 + 1  # conv stack formula
            out_lengths = out_lengths.long().clamp(min=1, max=features.shape[1])
        else:
            T_prime = features.shape[1]
            out_lengths = torch.full(
                (waveform.shape[0],), T_prime,
                dtype=torch.long, device=waveform.device,
            )

        return self.projection(features), out_lengths  # (B, T', model_size), (B,)


# ---------------------------------------------------------------------------
# SharedTransformer
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SharedTransformer(nn.Module):
    """
    Standard PyTorch Transformer encoder (batch_first) with sinusoidal
    positional encoding. batch_first=True + no key_padding_mask lets
    scaled_dot_product_attention dispatch to FlashAttention on Ampere/Hopper.

    Input:  (B, T, model_size)
    Output: (B, T, model_size)
    """

    def __init__(
        self,
        model_size:      int   = 256,
        num_layers:      int   = 4,
        nhead:           int   = 8,
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
        max_len:         int   = 4096,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(model_size, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.transformer(x)


# ---------------------------------------------------------------------------
# CTCHead
# ---------------------------------------------------------------------------

class CTCHead(nn.Module):
    """
    Linear projection + log-softmax.

    Call forward() to get log-probs for CTC decoding.
    Call compute_ctc_loss() to get the scalar CTC loss.
    """

    def __init__(self, model_size: int, vocab_size: int):
        """
        Args:
            vocab_size: total output classes INCLUDING the CTC blank token.
        """
        super().__init__()
        self.linear = nn.Linear(model_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, model_size)
        Returns:
            log_probs: (B, T, vocab_size)  — always fp32 for CTC numerics
        """
        return F.log_softmax(self.linear(x).float(), dim=-1)

    def compute_ctc_loss(
        self,
        log_probs: torch.Tensor,        # (B, T, vocab_size)
        targets: torch.Tensor,           # (B, L)  padded
        input_lengths: torch.Tensor,     # (B,)
        target_lengths: torch.Tensor,    # (B,)
        blank: int = 0,
    ) -> torch.Tensor:
        """Compute mean CTC loss over the batch."""
        # F.ctc_loss expects (T, B, C)
        log_probs_t = log_probs.transpose(0, 1).contiguous()
        return F.ctc_loss(
            log_probs_t,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction='mean',
            zero_infinity=True,
        )


# ---------------------------------------------------------------------------
# BaselineModel
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """
    EMGEncoder → SharedTransformer → CTCHead

    All parameters are trainable.
    """

    def __init__(
        self,
        vocab_size:  int,
        model_size:  int   = 768,
        num_layers:  int   = 6,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.encoder     = EMGEncoder(model_size=model_size)
        self.transformer = SharedTransformer(
            model_size=model_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.ctc_head    = CTCHead(model_size=model_size, vocab_size=vocab_size)
        self.blank_id    = vocab_size - 1  # blank is the last index

    def forward(
        self,
        raw_emg: torch.Tensor,           # (B, T_raw, 8)
        return_loss: bool = False,
        targets: torch.Tensor | None = None,          # (B, L)
        input_lengths: torch.Tensor | None = None,    # (B,) — EMG frame counts
        target_lengths: torch.Tensor | None = None,   # (B,)
    ) -> dict:
        """
        Returns a dict with:
            log_probs  : (B, T, vocab_size)
            loss       : scalar (only when return_loss=True)
            enc_lengths: (B,) — T values after the 8× stride
        """
        x = self.encoder(raw_emg)           # (B, T, model_size)
        x = self.transformer(x)              # (B, T, model_size)
        log_probs = self.ctc_head(x)         # (B, T, vocab_size)

        # Downsampled lengths: T_raw // 8
        T_raw = raw_emg.shape[1]
        enc_lengths = torch.full(
            (raw_emg.shape[0],), T_raw // 8,
            dtype=torch.long, device=raw_emg.device
        )
        if input_lengths is not None:
            # Respect per-sample lengths (accounting for padding)
            enc_lengths = (input_lengths.float() / 8).floor().long().clamp(min=1)

        out = {'log_probs': log_probs, 'enc_lengths': enc_lengths}

        if return_loss:
            assert targets is not None and target_lengths is not None
            out['loss'] = self.ctc_head.compute_ctc_loss(
                log_probs, targets, enc_lengths, target_lengths,
                blank=self.blank_id,
            )
        return out


# ---------------------------------------------------------------------------
# UMLModel
# ---------------------------------------------------------------------------

class UMLModel(nn.Module):
    """
    Dual-branch model sharing a single Transformer.

    EMG  branch: EMGEncoder  → SharedTransformer → CTCHead
    Audio branch: AudioEncoder → SharedTransformer → CTCHead

    The SharedTransformer weights are shared (same Python object).
    AudioEncoder is always frozen.
    At inference, only the EMG branch is used.
    """

    def __init__(
        self,
        vocab_size:  int,
        model_size:  int   = 768,
        num_layers:  int   = 6,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.emg_encoder   = EMGEncoder(model_size=model_size)
        self.audio_encoder = AudioEncoder(model_size=model_size)
        self.transformer   = SharedTransformer(          # shared by both branches
            model_size=model_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)
        self.blank_id = vocab_size - 1

    # ------------------------------------------------------------------
    # EMG branch (used at training and inference)
    # ------------------------------------------------------------------

    def forward_emg(
        self,
        raw_emg: torch.Tensor,          # (B, T_raw, 8)
        return_loss: bool = False,
        targets: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict:
        x = self.emg_encoder(raw_emg)   # (B, T, model_size)
        x = self.transformer(x)
        log_probs = self.ctc_head(x)

        T_raw = raw_emg.shape[1]
        enc_lengths = torch.full(
            (raw_emg.shape[0],), T_raw // 8,
            dtype=torch.long, device=raw_emg.device
        )
        if input_lengths is not None:
            enc_lengths = (input_lengths.float() / 8).floor().long().clamp(min=1)

        out = {'log_probs': log_probs, 'enc_lengths': enc_lengths}
        if return_loss:
            assert targets is not None and target_lengths is not None
            out['loss'] = self.ctc_head.compute_ctc_loss(
                log_probs, targets, enc_lengths, target_lengths,
                blank=self.blank_id,
            )
        return out

    # ------------------------------------------------------------------
    # Audio branch (training-only auxiliary)
    # ------------------------------------------------------------------

    def forward_audio(
        self,
        waveform: torch.Tensor,          # (B, T_audio)  zero-padded
        targets: torch.Tensor,            # (B, L)
        target_lengths: torch.Tensor,     # (B,)
        audio_lengths: torch.Tensor | None = None,  # (B,) real waveform lengths
    ) -> dict:
        """
        AudioEncoder is always frozen (wav2vec2 inside uses torch.no_grad()).
        audio_lengths is forwarded so wav2vec2 builds a proper attention_mask.
        """
        x, enc_lengths = self.audio_encoder(waveform, audio_lengths)  # (B, T', model_size), (B,)
        x = self.transformer(x)
        log_probs = self.ctc_head(x)   # (B, T', vocab_size)

        loss = self.ctc_head.compute_ctc_loss(
            log_probs, targets, enc_lengths, target_lengths,
            blank=self.blank_id,
        )
        return {'log_probs': log_probs, 'enc_lengths': enc_lengths, 'loss': loss}

    # ------------------------------------------------------------------
    # Audio branch — from precomputed wav2vec2 features (fast path)
    # ------------------------------------------------------------------

    def forward_audio_from_features(
        self,
        features: torch.Tensor,          # (B, T', 768) — wav2vec2 last_hidden_state
        feat_lengths: torch.Tensor,       # (B,) valid frames in T'
        targets: torch.Tensor,            # (B, L)
        target_lengths: torch.Tensor,     # (B,)
    ) -> dict:
        """
        Skip the frozen wav2vec2 forward entirely — apply only the trainable
        projection + SharedTransformer + CTCHead on precomputed features.
        Equivalent to forward_audio() when the cache was built from the same
        wav2vec2 checkpoint.
        """
        x = self.audio_encoder.projection(features)   # (B, T', model_size)
        x = self.transformer(x)
        log_probs = self.ctc_head(x)

        enc_lengths = feat_lengths.long().clamp(min=1, max=features.shape[1])
        loss = self.ctc_head.compute_ctc_loss(
            log_probs, targets, enc_lengths, target_lengths,
            blank=self.blank_id,
        )
        return {'log_probs': log_probs, 'enc_lengths': enc_lengths, 'loss': loss}

    # ------------------------------------------------------------------
    # Convenience: inference uses EMG branch only
    # ------------------------------------------------------------------

    def forward(self, raw_emg: torch.Tensor, **kwargs) -> dict:
        return self.forward_emg(raw_emg, **kwargs)
