"""Comprehensive feature extraction for Voice Biomarker Platform v2."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required: pip install librosa")

try:
    import parselmouth
except ImportError:
    parselmouth = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract acoustic and biomarker-specific features from audio signals.

    Provides methods for general acoustic features (MFCCs, pitch, HNR,
    formants), voice quality measures (jitter, shimmer), and disease-
    specific feature sets for Parkinson's disease and depression screening.

    Example::

        fe = FeatureExtractor()
        features = fe.extract_all(y, sr=22050)
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        hop_length: int = 512,
        n_fft: int = 2048,
        fmin: float = 50.0,
        fmax: float = 5000.0,
        n_mels: int = 128,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels

    # ------------------------------------------------------------------
    # Voice quality (Parselmouth / Praat)
    # ------------------------------------------------------------------

    def _get_sound(self, y: np.ndarray, sr: int):
        """Return a parselmouth.Sound object."""
        if parselmouth is None:
            raise ImportError("parselmouth is required for jitter/shimmer: pip install parselmouth")
        return parselmouth.Sound(y, sampling_frequency=sr)

    def extract_jitter_shimmer(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract jitter and shimmer voice quality measures.

        Uses Praat via parselmouth when available; falls back to a librosa-
        based F0 estimation otherwise.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary of jitter and shimmer values.
        """
        if parselmouth is not None:
            return self._extract_jitter_shimmer_praat(y, sr)
        return self._extract_jitter_shimmer_librosa(y, sr)

    def _extract_jitter_shimmer_praat(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Praat-based jitter/shimmer extraction."""
        sound = self._get_sound(y, sr)
        pitch = sound.to_pitch_ac(time_step=1 / sr, pitch_floor=self.fmin, pitch_ceiling=self.fmax)

        point_process = sound.to_pointprocess_cc(
            time_step=1 / sr, pitch_floor=self.fmin, pitch_ceiling=self.fmax
        )

        result: Dict[str, float] = {}

        try:
            result["jitter_local"] = float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        except Exception:
            result["jitter_local"] = 0.0

        try:
            result["jitter_rap"] = float(parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        except Exception:
            result["jitter_rap"] = 0.0

        try:
            result["jitter_ppq5"] = float(parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
        except Exception:
            result["jitter_ppq5"] = 0.0

        try:
            result["shimmer_local"] = float(parselmouth.praat.call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except Exception:
            result["shimmer_local"] = 0.0

        try:
            result["shimmer_apq3"] = float(parselmouth.praat.call(point_process, "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except Exception:
            result["shimmer_apq3"] = 0.0

        try:
            result["shimmer_apq5"] = float(parselmouth.praat.call(point_process, "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except Exception:
            result["shimmer_apq5"] = 0.0

        logger.info("Jitter/Shimmer extracted: jitter_local=%.4f, shimmer_local=%.4f",
                     result.get("jitter_local", 0), result.get("shimmer_local", 0))
        return result

    def _extract_jitter_shimmer_librosa(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Librosa-based fallback for jitter/shimmer when parselmouth is unavailable."""
        f0, voiced_flag, _ = librosa.pyin(y, fmin=self.fmin, fmax=self.fmax, sr=sr)
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_valid) < 2:
            return {
                "jitter_local": 0.0, "jitter_rap": 0.0, "jitter_ppq5": 0.0,
                "shimmer_local": 0.0, "shimmer_apq3": 0.0, "shimmer_apq5": 0.0,
            }

        periods = 1.0 / f0_valid
        period_diffs = np.abs(np.diff(periods))
        mean_period = np.mean(periods)

        jitter_local = float(np.mean(period_diffs) / mean_period) if mean_period > 0 else 0.0

        jitter_rap = 0.0
        if len(periods) >= 3:
            rap_diffs = np.abs(periods[1:-1] - (periods[:-2] + periods[2:]) / 2)
            jitter_rap = float(np.mean(rap_diffs) / mean_period)

        jitter_ppq5 = 0.0
        if len(periods) >= 5:
            kernel = np.ones(5) / 5
            ppq5_smoothed = np.convolve(periods, kernel, mode="valid")
            n = min(len(periods) - 4, len(ppq5_smoothed))
            ppq5_diffs = np.abs(periods[2:2 + n] - ppq5_smoothed[:n])
            jitter_ppq5 = float(np.mean(ppq5_diffs) / mean_period) if mean_period > 0 else 0.0

        hop_length = 512
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        voiced_indices = np.where(voiced_flag)[0]

        shimmer_local = 0.0
        shimmer_apq3 = 0.0
        shimmer_apq5 = 0.0

        if len(voiced_indices) >= 2:
            voiced_amps = np.mean(S[:, voiced_indices], axis=0)
            amp_diffs = np.abs(np.diff(voiced_amps))
            mean_amp = np.mean(voiced_amps)
            shimmer_local = float(np.mean(amp_diffs) / mean_amp) if mean_amp > 0 else 0.0

            if len(voiced_amps) >= 3:
                apq3_diffs = np.abs(voiced_amps[1:-1] - (voiced_amps[:-2] + voiced_amps[2:]) / 2)
                shimmer_apq3 = float(np.mean(apq3_diffs) / mean_amp) if mean_amp > 0 else 0.0
            else:
                shimmer_apq3 = shimmer_local

            if len(voiced_amps) >= 5:
                apq5_smoothed = np.convolve(voiced_amps, np.ones(5) / 5, mode="valid")
                n = min(len(voiced_amps) - 4, len(apq5_smoothed))
                apq5_diffs = np.abs(voiced_amps[2:2 + n] - apq5_smoothed[:n])
                shimmer_apq5 = float(np.mean(apq5_diffs) / mean_amp) if mean_amp > 0 else 0.0
            else:
                shimmer_apq5 = shimmer_local

        return {
            "jitter_local": float(jitter_local),
            "jitter_rap": float(jitter_rap),
            "jitter_ppq5": float(jitter_ppq5),
            "shimmer_local": float(shimmer_local),
            "shimmer_apq3": float(shimmer_apq3),
            "shimmer_apq5": float(shimmer_apq5),
        }

    # ------------------------------------------------------------------
    # HNR
    # ------------------------------------------------------------------

    def extract_hnr(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract Harmonics-to-Noise Ratio.

        Uses Parselmouth to compute HNR, which reflects the ratio of
        periodic to aperiodic energy in the signal.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary with ``hnr_db`` value.
        """
        if parselmouth is not None:
            try:
                sound = self._get_sound(y, sr)
                harmonicity = sound.to_harmonicity_ac(time_step=1 / sr, pitch_floor=self.fmin)
                hnr = float(harmonicity.get_mean())
                return {"hnr_db": round(hnr, 4)}
            except Exception as exc:
                logger.warning("Parselmouth HNR failed, using spectral fallback: %s", exc)

        return self._hnr_spectral_fallback(y, sr)

    def _hnr_spectral_fallback(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Estimate HNR from the power spectrum."""
        S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)

        f0_est = self._estimate_f0_for_hnr(y, sr)
        if f0_est < 1:
            return {"hnr_db": 0.0}

        harmonic_energy = 0.0
        n_harmonics = 0
        for k in range(1, int(sr / (2 * f0_est)) + 1):
            target_freq = f0_est * k
            idx = np.argmin(np.abs(freqs - target_freq))
            if idx < len(S):
                harmonic_energy += float(S[idx]) ** 2
                n_harmonics += 1

        total_energy = float(np.sum(S ** 2)) + 1e-12
        noise_energy = max(total_energy - harmonic_energy, 1e-12)
        hnr = 10.0 * np.log10(max(harmonic_energy, 1e-12) / noise_energy)
        return {"hnr_db": round(float(hnr), 4)}

    @staticmethod
    def _estimate_f0_for_hnr(y: np.ndarray, sr: int) -> float:
        """Quick F0 estimate for HNR spectral fallback."""
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        return float(np.median(voiced)) if len(voiced) > 0 else 0.0

    # ------------------------------------------------------------------
    # MFCCs
    # ------------------------------------------------------------------

    def extract_mfcc(
        self,
        y: np.ndarray,
        sr: int,
        n_mfcc: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract MFCCs with delta and delta-delta coefficients.

        Returns the full coefficient matrix plus statistical summaries
        (mean, std, min, max) for each coefficient.

        Args:
            y: Audio signal.
            sr: Sample rate.
            n_mfcc: Number of MFCCs. ``None`` uses the instance default.

        Returns:
            Dictionary with ``mfcc``, ``mfcc_delta``, ``mfcc_delta2``
            arrays and ``mfcc_stats`` with per-coefficient statistics.
        """
        n = n_mfcc if n_mfcc is not None else self.n_mfcc

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n, hop_length=self.hop_length,
            n_fft=self.n_fft, fmin=self.fmin, fmax=self.fmax,
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        stats: Dict[str, List[float]] = {}
        for i in range(n):
            prefix = f"mfcc_{i}"
            stats[f"{prefix}_mean"] = [float(np.mean(mfcc[i]))]
            stats[f"{prefix}_std"] = [float(np.std(mfcc[i]))]
            stats[f"{prefix}_min"] = [float(np.min(mfcc[i]))]
            stats[f"{prefix}_max"] = [float(np.max(mfcc[i]))]

        return {
            "mfcc": mfcc,
            "mfcc_delta": mfcc_delta,
            "mfcc_delta2": mfcc_delta2,
            "mfcc_stats": stats,
        }

    # ------------------------------------------------------------------
    # Pitch (F0)
    # ------------------------------------------------------------------

    def extract_pitch(
        self,
        y: np.ndarray,
        sr: int,
        fmin: float = 50.0,
        fmax: float = 500.0,
    ) -> Dict[str, Any]:
        """Extract F0 pitch contour using the pYIN algorithm.

        Args:
            y: Audio signal.
            sr: Sample rate.
            fmin: Minimum pitch frequency.
            fmax: Maximum pitch frequency.

        Returns:
            Dictionary with ``f0`` array, ``f0_voiced`` (voiced only),
            ``f0_mean``, ``f0_std``, ``f0_min``, ``f0_max``, and
            ``voicing_fraction``.
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr,
            hop_length=self.hop_length,
        )

        f0_arr = np.asarray(f0, dtype=float)
        voiced_mask = ~np.isnan(f0_arr)
        f0_voiced = f0_arr[voiced_mask]

        result: Dict[str, Any] = {
            "f0": f0_arr,
            "f0_voiced": f0_voiced,
        }

        if len(f0_voiced) > 0:
            result["f0_mean"] = round(float(np.mean(f0_voiced)), 2)
            result["f0_std"] = round(float(np.std(f0_voiced)), 2)
            result["f0_min"] = round(float(np.min(f0_voiced)), 2)
            result["f0_max"] = round(float(np.max(f0_voiced)), 2)
        else:
            result["f0_mean"] = 0.0
            result["f0_std"] = 0.0
            result["f0_min"] = 0.0
            result["f0_max"] = 0.0

        result["voicing_fraction"] = round(float(np.mean(voiced_mask)), 4)
        return result

    # ------------------------------------------------------------------
    # Pause patterns
    # ------------------------------------------------------------------

    def extract_pause_patterns(
        self,
        y: np.ndarray,
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        silence_db_threshold: float = -40.0,
        min_pause_duration: float = 0.15,
    ) -> Dict[str, Any]:
        """Detect pauses and compute pause statistics.

        Args:
            y: Audio signal.
            sr: Sample rate.
            frame_length: Analysis frame length.
            hop_length: Hop length.
            silence_db_threshold: Amplitude threshold (dB) for silence.
            min_pause_duration: Minimum pause duration to count (seconds).

        Returns:
            Dictionary with pause statistics: ``n_pauses``, ``pause_ratio``,
            ``mean_pause_duration``, ``total_pause_duration``, etc.
        """
        frame_energy = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        frame_db = librosa.amplitude_to_db(frame_energy, ref=np.max)
        silence_mask = frame_db < silence_db_threshold

        pause_durations: List[float] = []
        in_pause = False
        pause_start = 0

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                dur = (i - pause_start) * hop_length / sr
                if dur >= min_pause_duration:
                    pause_durations.append(dur)

        if in_pause:
            dur = (len(silence_mask) - pause_start) * hop_length / sr
            if dur >= min_pause_duration:
                pause_durations.append(dur)

        total_duration = len(y) / sr
        total_pause = sum(pause_durations)

        result: Dict[str, Any] = {
            "n_pauses": len(pause_durations),
            "pause_ratio": round(total_pause / max(total_duration, 1e-6), 4),
            "total_pause_duration": round(total_pause, 4),
            "mean_pause_duration": round(float(np.mean(pause_durations)), 4) if pause_durations else 0.0,
            "max_pause_duration": round(float(np.max(pause_durations)), 4) if pause_durations else 0.0,
            "min_pause_duration": round(float(np.min(pause_durations)), 4) if pause_durations else 0.0,
            "std_pause_duration": round(float(np.std(pause_durations)), 4) if len(pause_durations) > 1 else 0.0,
        }
        return result

    # ------------------------------------------------------------------
    # Speech rate
    # ------------------------------------------------------------------

    def extract_speech_rate(
        self,
        y: np.ndarray,
        sr: int,
        onset_env_threshold: float = 0.05,
    ) -> Dict[str, float]:
        """Estimate speech rate metrics from spectral onsets.

        Uses onset detection as a proxy for syllable-like events.

        Args:
            y: Audio signal.
            sr: Sample rate.
            onset_env_threshold: Normalized threshold for onset detection.

        Returns:
            Dictionary with ``speech_rate`` (events/sec),
            ``articulation_rate`` (events/sec excluding pauses),
            and ``total_duration``.
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length,
            onset_envelope=onset_env,
            backtrack=False,
        )

        total_duration = len(y) / sr

        pause_data = self.extract_pause_patterns(y, sr)
        pause_duration = pause_data["total_pause_duration"]
        speech_duration = max(total_duration - pause_duration, 1e-6)

        n_events = len(onsets)

        return {
            "speech_rate": round(n_events / max(total_duration, 1e-6), 4),
            "articulation_rate": round(n_events / speech_duration, 4),
            "n_onset_events": n_events,
            "total_duration": round(total_duration, 4),
            "speech_duration": round(speech_duration, 4),
        }

    # ------------------------------------------------------------------
    # Formants
    # ------------------------------------------------------------------

    def extract_formants(
        self,
        y: np.ndarray,
        sr: int,
        n_formants: int = 3,
        win_length: int = 0.025,
    ) -> Dict[str, Any]:
        """Extract formant frequencies F1, F2, F3 using LPC analysis.

        Args:
            y: Audio signal.
            sr: Sample rate.
            n_formants: Number of formants to extract.
            win_length: Analysis window length in seconds.

        Returns:
            Dictionary with ``f1_mean``, ``f2_mean``, ``f3_mean`` and
            per-formant statistics.
        """
        if parselmouth is not None:
            try:
                return self._formants_parselmouth(y, sr, n_formants, win_length)
            except Exception as exc:
                logger.warning("Parselmouth formant extraction failed, using librosa fallback: %s", exc)

        return self._formants_librosa_fallback(y, sr, n_formants)

    def _formants_parselmouth(
        self,
        y: np.ndarray,
        sr: int,
        n_formants: int,
        win_length: float,
    ) -> Dict[str, Any]:
        """Formant extraction via Parselmouth/Formant tracking."""
        sound = self._get_sound(y, sr)
        formant = sound.to_formant_burg(
            time_step=1 / sr, max_number_of_formants=n_formants,
            maximum_formant=5500, window_length=win_length,
            pre_emphasis_from=50,
        )

        result: Dict[str, Any] = {}
        all_formants: Dict[int, List[float]] = {i + 1: [] for i in range(n_formants)}

        n_frames = formant.get_number_of_frames()
        for fi in range(1, n_formants + 1):
            for ti in range(n_frames):
                val = formant.get_value_at_number_of_formants(fi, ti + 1)
                if not np.isnan(val):
                    all_formants[fi].append(val)

        for fi in range(1, n_formants + 1):
            key = f"f{fi}"
            vals = all_formants[fi]
            if vals:
                result[f"{key}_mean"] = round(float(np.mean(vals)), 2)
                result[f"{key}_std"] = round(float(np.std(vals)), 2)
                result[f"{key}_min"] = round(float(np.min(vals)), 2)
                result[f"{key}_max"] = round(float(np.max(vals)), 2)
            else:
                result[f"{key}_mean"] = 0.0
                result[f"{key}_std"] = 0.0
                result[f"{key}_min"] = 0.0
                result[f"{key}_max"] = 0.0

        return result

    def _formants_librosa_fallback(
        self,
        y: np.ndarray,
        sr: int,
        n_formants: int,
    ) -> Dict[str, Any]:
        """Estimate formants via spectral peak picking."""
        from scipy.signal import find_peaks

        S = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)

        peak_indices, _ = find_peaks(S, distance=10, prominence=np.max(S) * 0.05)
        peak_freqs = np.sort(freqs[peak_indices])

        vocal_mask = (peak_freqs > 200) & (peak_freqs < 5000)
        vocal_peaks = peak_freqs[vocal_mask]

        result: Dict[str, float] = {}
        for i in range(min(n_formants, len(vocal_peaks))):
            result[f"f{i + 1}_mean"] = round(float(vocal_peaks[i]), 2)
            result[f"f{i + 1}_std"] = 0.0
            result[f"f{i + 1}_min"] = round(float(vocal_peaks[i]), 2)
            result[f"f{i + 1}_max"] = round(float(vocal_peaks[i]), 2)

        for i in range(len(vocal_peaks) + 1, n_formants + 1):
            result[f"f{i}_mean"] = 0.0
            result[f"f{i}_std"] = 0.0
            result[f"f{i}_min"] = 0.0
            result[f"f{i}_max"] = 0.0

        return result

    # ------------------------------------------------------------------
    # Prosodic / spectral features
    # ------------------------------------------------------------------

    def extract_prosodic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral and prosodic feature set.

        Includes spectral centroid, bandwidth, rolloff, contrast,
        flatness, zero-crossing rate, RMS energy, and mel spectrogram
        statistics.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary of prosodic feature values.
        """
        result: Dict[str, float] = {}

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        result["spectral_centroid_mean"] = round(float(np.mean(centroid)), 4)
        result["spectral_centroid_std"] = round(float(np.std(centroid)), 4)

        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        result["spectral_bandwidth_mean"] = round(float(np.mean(bw)), 4)
        result["spectral_bandwidth_std"] = round(float(np.std(bw)), 4)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        result["spectral_rolloff_mean"] = round(float(np.mean(rolloff)), 4)
        result["spectral_rolloff_std"] = round(float(np.std(rolloff)), 4)

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        for i in range(contrast.shape[0]):
            result[f"spectral_contrast_{i}_mean"] = round(float(np.mean(contrast[i])), 4)
            result[f"spectral_contrast_{i}_std"] = round(float(np.std(contrast[i])), 4)

        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        result["spectral_flatness_mean"] = round(float(np.mean(flatness)), 4)
        result["spectral_flatness_std"] = round(float(np.std(flatness)), 4)

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        result["zcr_mean"] = round(float(np.mean(zcr)), 6)
        result["zcr_std"] = round(float(np.std(zcr)), 6)

        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        result["rms_mean"] = round(float(np.mean(rms)), 6)
        result["rms_std"] = round(float(np.std(rms)), 6)
        result["rms_max"] = round(float(np.max(rms)), 6)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=self.hop_length,
            n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        result["mel_mean"] = round(float(np.mean(mel_db)), 4)
        result["mel_std"] = round(float(np.std(mel_db)), 4)

        return result

    # ------------------------------------------------------------------
    # Parkinson's-specific features
    # ------------------------------------------------------------------

    def extract_parkinsons_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract acoustic features relevant to Parkinson's disease.

        Combines jitter/shimmer measures, pitch variability, HNR,
        speech rate, and spectral features into a feature vector
        suitable for Parkinson's detection models.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary of Parkinson's-specific features.
        """
        result: Dict[str, float] = {}

        # Voice quality
        js = self.extract_jitter_shimmer(y, sr)
        for k, v in js.items():
            result[k] = round(v, 6)

        # HNR
        hnr = self.extract_hnr(y, sr)
        result.update(hnr)

        # Pitch variability
        pitch = self.extract_pitch(y, sr)
        for key in ("f0_mean", "f0_std", "f0_min", "f0_max", "voicing_fraction"):
            result[f"pd_{key}"] = pitch.get(key, 0.0)

        # Speech rate
        sr_data = self.extract_speech_rate(y, sr)
        for key in ("speech_rate", "articulation_rate"):
            result[f"pd_{key}"] = sr_data.get(key, 0.0)

        # Spectral features
        prosodic = self.extract_prosodic_features(y, sr)
        for key in ("spectral_centroid_mean", "spectral_bandwidth_mean",
                     "spectral_rolloff_mean", "spectral_flatness_mean",
                     "zcr_mean", "rms_mean"):
            result[f"pd_{key}"] = prosodic.get(key, 0.0)

        # Pause features
        pauses = self.extract_pause_patterns(y, sr)
        result["pd_pause_ratio"] = pauses.get("pause_ratio", 0.0)
        result["pd_n_pauses"] = float(pauses.get("n_pauses", 0))
        result["pd_mean_pause_duration"] = pauses.get("mean_pause_duration", 0.0)

        logger.info("Extracted %d Parkinson's features", len(result))
        return result

    # ------------------------------------------------------------------
    # Depression-specific features
    # ------------------------------------------------------------------

    def extract_depression_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract acoustic features relevant to depression screening.

        Focuses on prosodic flattening, spectral characteristics, pause
        patterns, and energy variability that are associated with
        depressive speech.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary of depression-specific features.
        """
        result: Dict[str, float] = {}

        # Pitch characteristics (monotony / flatness)
        pitch = self.extract_pitch(y, sr)
        f0_voiced = pitch.get("f0_voiced", np.array([]))
        result["dep_f0_mean"] = pitch.get("f0_mean", 0.0)
        result["dep_f0_std"] = pitch.get("f0_std", 0.0)
        result["dep_f0_range"] = pitch.get("f0_max", 0.0) - pitch.get("f0_min", 0.0)
        result["dep_voicing_fraction"] = pitch.get("voicing_fraction", 0.0)

        if len(f0_voiced) > 1:
            f0_diff = np.diff(f0_voiced)
            result["dep_f0_slope_mean"] = round(float(np.mean(np.abs(f0_diff))), 4)
            result["dep_f0_jitter"] = round(float(np.std(f0_diff) / (np.mean(np.abs(f0_diff)) + 1e-8)), 6)
        else:
            result["dep_f0_slope_mean"] = 0.0
            result["dep_f0_jitter"] = 0.0

        # Spectral flatness (flatness = monotony)
        prosodic = self.extract_prosodic_features(y, sr)
        result["dep_spectral_flatness"] = prosodic.get("spectral_flatness_mean", 0.0)
        result["dep_spectral_centroid"] = prosodic.get("spectral_centroid_mean", 0.0)
        result["dep_spectral_bandwidth"] = prosodic.get("spectral_bandwidth_mean", 0.0)
        result["dep_zcr"] = prosodic.get("zcr_mean", 0.0)

        # Energy / amplitude variation
        result["dep_rms_mean"] = prosodic.get("rms_mean", 0.0)
        result["dep_rms_std"] = prosodic.get("rms_std", 0.0)

        # Pause patterns (depression → longer pauses, higher pause ratio)
        pauses = self.extract_pause_patterns(y, sr)
        result["dep_pause_ratio"] = pauses.get("pause_ratio", 0.0)
        result["dep_n_pauses"] = float(pauses.get("n_pauses", 0))
        result["dep_mean_pause_duration"] = pauses.get("mean_pause_duration", 0.0)
        result["dep_max_pause_duration"] = pauses.get("max_pause_duration", 0.0)
        result["dep_std_pause_duration"] = pauses.get("std_pause_duration", 0.0)

        # Speech rate (depression → slower speech)
        sr_data = self.extract_speech_rate(y, sr)
        result["dep_speech_rate"] = sr_data.get("speech_rate", 0.0)
        result["dep_articulation_rate"] = sr_data.get("articulation_rate", 0.0)

        # MFCC summary (for timbre changes)
        mfcc_data = self.extract_mfcc(y, sr)
        mfcc_stats = mfcc_data.get("mfcc_stats", {})
        for k, v in mfcc_stats.items():
            if isinstance(v, list) and len(v) == 1:
                result[f"dep_{k}"] = v[0]

        logger.info("Extracted %d depression features", len(result))
        return result

    # ------------------------------------------------------------------
    # Aggregate extraction
    # ------------------------------------------------------------------

    def extract_all(
        self,
        y: np.ndarray,
        sr: int,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        """Extract all feature sets and return as a flat dictionary.

        Combines voice quality, prosodic, pitch, MFCC, pause, speech
        rate, formant, Parkinson's, and depression features into a
        single dictionary. Raw arrays (MFCC matrices, pitch contours)
        are excluded by default to keep the output scalar-friendly.

        Args:
            y: Audio signal.
            sr: Sample rate.
            include_raw: If ``True``, include raw arrays (MFCC matrices,
                pitch contour). Default ``False``.

        Returns:
            Dictionary of all extracted scalar features.
        """
        result: Dict[str, Any] = {}

        def _merge(data: Dict[str, Any]) -> None:
            for k, v in data.items():
                if include_raw or not isinstance(v, np.ndarray):
                    result[k] = v

        _merge(self.extract_jitter_shimmer(y, sr))
        _merge(self.extract_hnr(y, sr))

        mfcc_data = self.extract_mfcc(y, sr)
        _merge(mfcc_data.get("mfcc_stats", {}))
        if include_raw:
            result["mfcc"] = mfcc_data["mfcc"]
            result["mfcc_delta"] = mfcc_data["mfcc_delta"]
            result["mfcc_delta2"] = mfcc_data["mfcc_delta2"]

        pitch_data = self.extract_pitch(y, sr)
        for k, v in pitch_data.items():
            if k not in ("f0", "f0_voiced"):
                result[k] = v
            elif include_raw:
                result[k] = v

        _merge(self.extract_pause_patterns(y, sr))
        _merge(self.extract_speech_rate(y, sr))
        _merge(self.extract_formants(y, sr))
        _merge(self.extract_prosodic_features(y, sr))
        _merge(self.extract_parkinsons_features(y, sr))
        _merge(self.extract_depression_features(y, sr))

        logger.info("Extract_all complete: %d scalar features", len(result))
        return result

    # ------------------------------------------------------------------
    # Feature name registry
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return an ordered list of all scalar feature names.

        The list follows the same order as the feature groups in
        ``extract_all``: jitter/shimmer → HNR → MFCC stats → pitch →
        pauses → speech rate → formants → prosodic → Parkinson's →
        depression.

        Returns:
            Sorted, deduplicated list of feature name strings.
        """
        names: List[str] = []

        for prefix in ("jitter_local", "jitter_rap", "jitter_ppq5",
                        "shimmer_local", "shimmer_apq3", "shimmer_apq5"):
            names.append(prefix)

        names.append("hnr_db")

        for i in range(self.n_mfcc):
            for suffix in ("mean", "std", "min", "max"):
                names.append(f"mfcc_{i}_{suffix}")

        for key in ("f0_mean", "f0_std", "f0_min", "f0_max", "voicing_fraction"):
            names.append(key)

        for key in ("n_pauses", "pause_ratio", "total_pause_duration",
                     "mean_pause_duration", "max_pause_duration",
                     "min_pause_duration", "std_pause_duration"):
            names.append(key)

        for key in ("speech_rate", "articulation_rate", "n_onset_events",
                     "total_duration", "speech_duration"):
            names.append(key)

        for i in range(1, 4):
            for suffix in ("mean", "std", "min", "max"):
                names.append(f"f{i}_{suffix}")

        for key in ("spectral_centroid_mean", "spectral_centroid_std",
                     "spectral_bandwidth_mean", "spectral_bandwidth_std",
                     "spectral_rolloff_mean", "spectral_rolloff_std"):
            names.append(key)

        for i in range(7):
            names.append(f"spectral_contrast_{i}_mean")
            names.append(f"spectral_contrast_{i}_std")

        for key in ("spectral_flatness_mean", "spectral_flatness_std",
                     "zcr_mean", "zcr_std", "rms_mean", "rms_std", "rms_max",
                     "mel_mean", "mel_std"):
            names.append(key)

        pd_prefixes = (
            "pd_f0_mean", "pd_f0_std", "pd_f0_min", "pd_f0_max", "pd_voicing_fraction",
            "pd_speech_rate", "pd_articulation_rate",
            "pd_spectral_centroid_mean", "pd_spectral_bandwidth_mean",
            "pd_spectral_rolloff_mean", "pd_spectral_flatness_mean",
            "pd_zcr_mean", "pd_rms_mean",
            "pd_pause_ratio", "pd_n_pauses", "pd_mean_pause_duration",
        )
        names.extend(pd_prefixes)

        dep_prefixes = (
            "dep_f0_mean", "dep_f0_std", "dep_f0_range", "dep_voicing_fraction",
            "dep_f0_slope_mean", "dep_f0_jitter",
            "dep_spectral_flatness", "dep_spectral_centroid", "dep_spectral_bandwidth",
            "dep_zcr", "dep_rms_mean", "dep_rms_std",
            "dep_pause_ratio", "dep_n_pauses", "dep_mean_pause_duration",
            "dep_max_pause_duration", "dep_std_pause_duration",
            "dep_speech_rate", "dep_articulation_rate",
        )
        names.extend(dep_prefixes)

        for i in range(self.n_mfcc):
            for suffix in ("mean", "std", "min", "max"):
                names.append(f"dep_mfcc_{i}_{suffix}")

        seen = set()
        unique: List[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique
