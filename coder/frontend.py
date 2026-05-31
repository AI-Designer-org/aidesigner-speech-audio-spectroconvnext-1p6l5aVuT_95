"""
SpectroConvNeXt — Audio frontend for log-mel spectrogram extraction.

Pipeline:
    Raw audio (arbitrary sample rate)
    → Resample to 22050 Hz
    → Trim/pad to 2.0 seconds
    → STFT → Mel filterbank → Log compression
    → Per-sample mean/std normalization
    → Shape: (B, 1, 128, T_frames)

Typical output shape: (B, 1, 128, 86) at 22050 Hz with n_fft=1024, hop_length=512.
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Optional


class AudioFrontend(nn.Module):
    """
    Audio feature extraction: waveform → normalized log-mel spectrogram.

    Args:
        sample_rate:       Target sample rate in Hz (default: 22050)
        clip_duration_sec: Duration in seconds (default: 2.0)
        n_fft:             FFT size (default: 1024)
        hop_length:        Hop length in samples (default: 512)
        n_mels:            Number of mel filterbanks (default: 128)
        power:             Power of the spectrogram (2.0 = power spectrogram)
        f_min:             Minimum frequency in Hz (default: 0)
        f_max:             Maximum frequency in Hz (default: None = nyquist)
        log_offset:        Small constant added before log to avoid log(0)
        normalize:         Whether to apply per-sample mean/std normalization
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        clip_duration_sec: float = 2.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        power: float = 2.0,
        f_min: int = 0,
        f_max: Optional[int] = None,
        log_offset: float = 1e-6,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_duration_sec)  # e.g. 44100
        self.log_offset = log_offset
        self.normalize = normalize

        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate // 2,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveform to normalized log-mel spectrogram.

        Args:
            waveform: (B, 1, T_samples) — raw audio signals.
                      Values should be in [-1, 1] range.

        Returns:
            (B, 1, n_mels, T_frames) — log-mel spectrogram.
            T_frames ≈ ceil(T_samples / hop_length).
            Typical shape at 22050 Hz: (B, 1, 128, 86)
        """
        B = waveform.shape[0]

        # ── Resample to target sample rate (if needed) ───────────────────
        # If the input sample rate differs, user must resample externally.
        # We assume input is at self.sample_rate already.

        # ── Trim or pad to exact length ──────────────────────────────────
        if waveform.shape[-1] > self.target_samples:
            waveform = waveform[..., :self.target_samples]           # trim  (B, 1, T_target)
        elif waveform.shape[-1] < self.target_samples:
            pad_len = self.target_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(
                waveform, (0, pad_len), mode="constant", value=0.0,
            )                                                         # pad   (B, 1, T_target)

        # ── Compute mel spectrogram ─────────────────────────────────────
        # torchaudio MelSpectrogram expects (B, T) or (B, 1, T)
        mel = self.mel_spec(waveform)                                 # (B, n_mels, T_frames)

        # ── Log compression ─────────────────────────────────────────────
        mel = torch.log(mel + self.log_offset)                        # (B, n_mels, T_frames)

        # ── Per-sample mean/std normalization ───────────────────────────
        if self.normalize:
            mean = mel.mean(dim=(1, 2), keepdim=True)                 # (B, 1, 1)
            std = mel.std(dim=(1, 2), keepdim=True) + 1e-6           # (B, 1, 1)
            mel = (mel - mean) / std                                  # (B, n_mels, T_frames)

        # ── Add channel dimension (1 for mono) ─────────────────────────
        mel = mel.unsqueeze(1)                                         # (B, 1, n_mels, T_frames)

        return mel

    def get_num_frames(self, num_samples: Optional[int] = None) -> int:
        """
        Compute the number of time frames given a number of audio samples.

        Args:
            num_samples: Number of audio samples. If None, uses target_samples.

        Returns:
            Number of time frames in the spectrogram.
        """
        n = num_samples if num_samples is not None else self.target_samples
        return 1 + (n - self.mel_spec.n_fft) // self.mel_spec.hop_length
