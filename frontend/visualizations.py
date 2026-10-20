"""Plotting utilities for the Voice Biomarker Platform."""

from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Audio visualizations
# ---------------------------------------------------------------------------


def plot_waveform(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Audio Waveform",
) -> go.Figure:
    """Create an interactive audio waveform plot.

    Args:
        audio: 1-D audio samples (int16 or float).
        sr: Sample rate in Hz.
        title: Plot title.

    Returns:
        A Plotly ``Figure`` with the waveform.
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, num=len(audio))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=audio,
            mode="lines",
            line=dict(width=0.5, color="#1f77b4"),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.3)",
            name="Amplitude",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=30),
        height=300,
    )
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Spectrogram",
    n_fft: int = 1024,
    hop_length: int = 512,
) -> go.Figure:
    """Create a standard spectrogram using short-time FFT.

    Args:
        audio: 1-D audio samples.
        sr: Sample rate in Hz.
        title: Plot title.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        A Plotly ``Figure``.
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Compute STFT magnitude
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.stack(
        [audio[i * hop_length : i * hop_length + n_fft] for i in range(max(n_frames, 1))]
    )
    windowed = frames * window[np.newaxis, :]
    spectrum = np.abs(np.fft.rfft(windowed, axis=-1))
    log_spectrum = 20.0 * np.log10(spectrum + 1e-10)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(n_frames) * hop_length / sr

    fig = go.Figure(
        data=go.Heatmap(
            z=log_spectrum.T,
            x=times,
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=30),
        height=350,
    )
    return fig


def plot_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Mel-Spectrogram",
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> go.Figure:
    """Create a mel-spectrogram visualization.

    Args:
        audio: 1-D audio samples.
        sr: Sample rate in Hz.
        title: Plot title.
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        A Plotly ``Figure``.
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Build mel filterbank manually
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_low = hz_to_mel(0)
    mel_high = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.round(hz_points / (sr / n_fft)).astype(int)

    n_freqs = n_fft // 2 + 1
    mel_fb = np.zeros((n_mels, n_freqs))
    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(left, min(center, n_freqs)):
            mel_fb[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, min(right, n_freqs)):
            mel_fb[i - 1, j] = (right - j) / max(right - center, 1)

    # STFT
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.stack(
        [audio[i * hop_length : i * hop_length + n_fft] for i in range(max(n_frames, 1))]
    )
    windowed = frames * window[np.newaxis, :]
    magnitude = np.abs(np.fft.rfft(windowed, axis=-1))
    mel_spec = mel_fb @ magnitude.T
    log_mel = 20.0 * np.log10(mel_spec + 1e-10)

    times = np.arange(n_frames) * hop_length / sr
    mel_axis = np.arange(n_mels)

    fig = go.Figure(
        data=go.Heatmap(
            z=log_mel,
            x=times,
            y=mel_axis,
            colorscale="Magma",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Mel Band",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=30),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# SHAP / Explainability
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values: dict[str, float],
    title: str = "SHAP Feature Contribution (Summary)",
) -> go.Figure:
    """Horizontal bar chart of SHAP values for each feature.

    Args:
        shap_values: Mapping of feature name to SHAP value.
        title: Plot title.

    Returns:
        A Plotly ``Figure``.
    """
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]))
    features = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#ff4b4b" if v > 0 else "#1f77b4" for v in values]

    fig = go.Figure(
        go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value",
        template="plotly_white",
        margin=dict(l=150, r=20, t=40, b=30),
        height=max(250, len(features) * 28),
    )
    return fig


def plot_shap_waterfall(
    shap_values: dict[str, float],
    base_value: float = 0.0,
    title: str = "SHAP Waterfall",
) -> go.Figure:
    """Waterfall chart for a single prediction's SHAP values.

    Args:
        shap_values: Mapping of feature name to SHAP value.
        base_value: Expected model output (base value).
        title: Plot title.

    Returns:
        A Plotly ``Figure``.
    """
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = ["base_value"] + [k for k, _ in sorted_items] + ["output"]
    cumulative = base_value
    vals: list[float] = [0.0]
    measures: list[str] = ["absolute"]
    texts: list[str] = [f"{base_value:.4f}"]

    for _, v in sorted_items:
        vals.append(v)
        cumulative += v
        measures.append("relative")
        texts.append(f"{v:+.4f}")

    vals.append(0.0)
    measures.append("total")
    texts.append(f"{cumulative:.4f}")

    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measures,
            x=labels,
            y=vals,
            text=texts,
            textposition="outside",
            connector=dict(line=dict(color="rgb(63,63,63)")),
            increasing=dict(marker=dict(color="#ff4b4b")),
            decreasing=dict(marker=dict(color="#1f77b4")),
            totals=dict(marker=dict(color="#2ca02c")),
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="Model output",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=30),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances: dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 20,
) -> go.Figure:
    """Bar chart of feature importances.

    Args:
        importances: Mapping of feature name to importance score.
        title: Plot title.
        top_n: Show only the top-N features.

    Returns:
        A Plotly ``Figure``.
    """
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [k for k, _ in sorted_items][::-1]
    values = [v for _, v in sorted_items][::-1]

    fig = go.Figure(
        go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker_color="#636EFA",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        template="plotly_white",
        margin=dict(l=180, r=20, t=40, b=30),
        height=max(300, len(features) * 28),
    )
    return fig


# ---------------------------------------------------------------------------
# Risk gauge
# ---------------------------------------------------------------------------


def plot_risk_gauge(
    risk_score: float,
    risk_level: str = "low",
    title: str = "Risk Score",
) -> go.Figure:
    """Gauge chart for risk score.

    Args:
        risk_score: Float between 0 and 1.
        risk_level: One of ``low``, ``medium``, ``high``.
        title: Plot title.

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            number={"suffix": "%"},
            title={"text": title, "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 33], "color": "#66bb6a"},
                    {"range": [33, 66], "color": "#ffa726"},
                    {"range": [66, 100], "color": "#ff4b4b"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": risk_score * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20))
    return fig


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Heatmap of a confusion matrix.

    Args:
        matrix: 2-D numpy array (n_classes x n_classes).
        labels: Class label names.
        title: Plot title.

    Returns:
        A Plotly ``Figure``.
    """
    if labels is None:
        labels = [str(i) for i in range(matrix.shape[0])]

    text = [[f"{val}" for val in row] for row in matrix]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Blues",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        height=400,
        width=450,
    )
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float = 0.0,
    title: str = "ROC Curve",
) -> go.Figure:
    """Plot an ROC curve.

    Args:
        fpr: False-positive rates.
        tpr: True-positive rates.
        auc_score: Area under the curve.
        title: Plot title.

    Returns:
        A Plotly ``Figure``.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc_score:.3f})",
            line=dict(width=2, color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(width=1, dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        yaxis=dict(range=[-0.02, 1.02]),
        height=400,
        width=450,
    )
    return fig
