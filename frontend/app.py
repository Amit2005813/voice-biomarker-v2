"""Voice Biomarker Platform v2 - Enhanced Streamlit Dashboard."""

import io
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

from frontend.visualizations import (
    plot_feature_importance,
    plot_mel_spectrogram,
    plot_risk_gauge,
    plot_shap_waterfall,
    plot_spectrogram,
    plot_waveform,
)

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Voice Biomarker Platform v2",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container { max-width: 1200px; padding-top: 2rem; }
    .stMetric { background: #f0f2f6; border-radius: 0.5rem; padding: 1rem; }
    div[data-testid="stSidebar"] { background-color: #f7f9fc; }
    .risk-high   { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa726; font-weight: bold; }
    .risk-low    { color: #66bb6a; font-weight: bold; }
    .header-container { padding: 1rem 0 2rem 0; }
    .header-container h1 { margin-bottom: 0.25rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULT_STATE: dict[str, Any] = {
    "api_url": "http://localhost:8000",
    "analysis_results": None,
    "history": [],
    "training_job_id": None,
    "selected_analysis": None,
    "waveform_data": None,
    "spectrogram_data": None,
    "request_counter": 0,
}

for key, default in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render sidebar navigation and return the selected page."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/microphone.png",
            width=64,
        )
        st.title("Voice Biomarker v2")
        st.divider()

        api_url = st.text_input(
            "API URL",
            value=st.session_state.api_url,
            help="Base URL of the backend API",
        )
        st.session_state.api_url = api_url.rstrip("/")

        st.divider()
        page = st.radio(
            "Navigation",
            [
                "Upload & Analyze",
                "Results History",
                "Visualizations",
                "Model Training",
                "About",
            ],
            label_visibility="collapsed",
        )
        st.divider()

        # Connection indicator
        try:
            resp = requests.get(f"{st.session_state.api_url}/health", timeout=3)
            if resp.status_code == 200:
                st.success("API Connected")
            else:
                st.warning("API Unhealthy")
        except requests.ConnectionError:
            st.error("API Unreachable")

    return page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_post(endpoint: str, **kwargs: Any) -> Optional[requests.Response]:
    """POST to the backend API with error handling."""
    try:
        resp = requests.post(
            f"{st.session_state.api_url}{endpoint}",
            timeout=300,
            **kwargs,
        )
        resp.raise_for_status()
        return resp
    except requests.ConnectionError:
        st.error("Cannot connect to the API. Please check the URL.")
    except requests.Timeout:
        st.error("Request timed out.")
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"API error ({exc.response.status_code}): {detail}")
    return None


def _api_get(endpoint: str, **kwargs: Any) -> Optional[requests.Response]:
    """GET from the backend API with error handling."""
    try:
        resp = requests.get(
            f"{st.session_state.api_url}{endpoint}",
            timeout=30,
            **kwargs,
        )
        resp.raise_for_status()
        return resp
    except requests.ConnectionError:
        st.error("Cannot connect to the API.")
    except requests.Timeout:
        st.error("Request timed out.")
    except requests.HTTPError as exc:
        st.error(f"API error: {exc}")
    return None


def _format_risk(risk_level: str) -> str:
    css_cls = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}.get(
        risk_level.lower(), ""
    )
    return f'<span class="{css_cls}">{risk_level.upper()}</span>'


# ---------------------------------------------------------------------------
# Page: Upload & Analyze
# ---------------------------------------------------------------------------


def page_upload_analyze() -> None:
    st.header("Upload & Analyze")
    st.markdown("Upload an audio file for voice biomarker analysis.")

    col_upload, col_player = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A",
        )

        if uploaded_file is not None:
            st.info(
                f"**File:** {uploaded_file.name}  \n"
                f"**Size:** {uploaded_file.size / 1024:.1f} KB  \n"
                f"**Type:** {uploaded_file.type}"
            )

            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes, format=uploaded_file.type or "audio/wav")

    with col_player:
        if uploaded_file is not None:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            fig = plot_waveform(audio_array, title=uploaded_file.name)
            st.plotly_chart(fig, use_container_width=True)

    if uploaded_file is not None:
        st.divider()
        if st.button("Run Analysis", type="primary", use_container_width=True):
            _run_analysis(uploaded_file)

    # Show latest results
    if st.session_state.analysis_results is not None:
        st.divider()
        _display_results(st.session_state.analysis_results)


def _run_analysis(uploaded_file: Any) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    progress_bar = st.progress(0, text="Uploading...")
    t0 = time.time()

    with st.spinner("Analyzing audio..."):
        progress_bar.progress(30, text="Sending to API...")
        resp = _api_post("/analyze", files=files)
        if resp is None:
            progress_bar.empty()
            return

        progress_bar.progress(70, text="Processing results...")
        result = resp.json()
        st.session_state.analysis_results = result
        st.session_state.history.append(result)

        progress_bar.progress(100, text=f"Done in {time.time() - t0:.1f}s")
        st.success("Analysis complete!")


def _display_results(result: dict) -> None:
    st.subheader("Analysis Results")

    # Extract risk from biomarkers if top-level risk_score is None
    risk_score = result.get("risk_score")
    biomarkers = result.get("biomarkers", {})
    if isinstance(biomarkers, dict):
        pk = biomarkers.get("parkinsons", {})
        dep = biomarkers.get("depression", {})
        if risk_score is None:
            risk_score = max(pk.get("risk_score", 0), dep.get("risk_score", 0))
    risk_score = risk_score or 0.0

    if risk_score >= 0.6:
        risk_level = "high"
    elif risk_score >= 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    analysis_id = result.get("analysis_id", "N/A")

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Level", risk_level.upper())
    m2.metric("Risk Score", f"{risk_score:.2%}")
    m3.metric("Analysis ID", str(analysis_id)[:12])

    # Risk gauge
    gauge_fig = plot_risk_gauge(risk_score, risk_level)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Show biomarker details
    if isinstance(biomarkers, dict) and biomarkers:
        st.subheader("Biomarker Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Parkinson's Indicators**")
            pd_data = biomarkers.get("parkinsons", {})
            if isinstance(pd_data, dict):
                st.metric("Risk", f"{pd_data.get('risk_score', 0):.1%}")
                st.metric("Level", pd_data.get("risk_level", "N/A"))

        with col2:
            st.markdown("**Depression Indicators**")
            dep_data = biomarkers.get("depression", {})
            if isinstance(dep_data, dict):
                st.metric("Risk", f"{dep_data.get('risk_score', 0):.1%}")
                st.metric("Level", dep_data.get("risk_level", "N/A"))

        # Anomalies
        anomalies = biomarkers.get("anomaly_details", [])
        if anomalies:
            st.subheader("Detected Anomalies")
            for a in anomalies:
                severity = a.get("severity", "unknown")
                color = "red" if severity in ("very_high", "high") else "orange" if severity == "medium" else "green"
                st.markdown(f"**{a.get('feature', 'N/A')}** ({a.get('condition', '')}) - "
                           f":{color}[{severity}] = {a.get('value', 'N/A')} "
                           f"(threshold: {a.get('threshold', 'N/A')})")
                st.caption(a.get("explanation", ""))

    # Raw JSON expander
    with st.expander("Raw JSON Response"):
        st.json(result)


# ---------------------------------------------------------------------------
# Page: Results History
# ---------------------------------------------------------------------------


def page_results_history() -> None:
    st.header("Results History")

    history = st.session_state.history
    if not history:
        st.info("No analyses performed yet. Go to **Upload & Analyze** to start.")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        date_filter = st.date_input(
            "Filter by date",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
        )
    with col_f2:
        risk_filter = st.multiselect(
            "Risk level", ["low", "medium", "high"], default=["low", "medium", "high"]
        )
    with col_f3:
        compare_count = st.number_input(
            "Compare top N", min_value=2, max_value=10, value=2
        )

    # Filter
    filtered = [
        r
        for r in history
        if r.get("risk_level", "").lower() in risk_filter
    ]

    if not filtered:
        st.warning("No results match the current filters.")
        return

    # Table
    rows = []
    for idx, r in enumerate(filtered):
        rows.append(
            {
                "Index": idx,
                "ID": r.get("analysis_id", "N/A"),
                "Risk": r.get("risk_level", "?").upper(),
                "Score": f'{r.get("risk_score", 0):.2%}',
                "Model": r.get("model_version", "?"),
            }
        )
    st.dataframe(rows, use_container_width=True)

    # Compare
    st.subheader("Compare Analyses")
    selected_indices = st.multiselect(
        "Select analyses to compare",
        list(range(len(filtered))),
        default=list(range(min(compare_count, len(filtered)))),
    )
    if len(selected_indices) >= 2:
        compare_analyses([filtered[i] for i in selected_indices])


def compare_analyses(analyses: list[dict]) -> None:
    """Display a side-by-side comparison of analyses."""
    cols = st.columns(len(analyses))
    for col, analysis in zip(cols, analyses):
        with col:
            st.markdown(
                f"**{analysis.get('analysis_id', 'N/A')}**  \n"
                f"Risk: {analysis.get('risk_level', '?').upper()}  |  "
                f"Score: {analysis.get('risk_score', 0):.2%}"
            )
            imp = analysis.get("feature_importances", {})
            if imp:
                fig = plot_feature_importance(imp)
                st.plotly_chart(fig, use_container_width=True, key=f"cmp_{analysis.get('analysis_id', id(analysis))}")


# ---------------------------------------------------------------------------
# Page: Visualizations
# ---------------------------------------------------------------------------


def page_visualizations() -> None:
    st.header("Visualizations")

    tab1, tab2, tab3 = st.tabs(["Upload for Viz", "From Results", "Demo"])

    with tab1:
        viz_file = st.file_uploader(
            "Upload audio for visualization",
            type=["wav", "mp3", "flac"],
            key="viz_upload",
        )
        if viz_file is not None:
            audio_bytes = viz_file.getvalue()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_waveform(audio_array, title="Waveform"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    plot_spectrogram(audio_array),
                    use_container_width=True,
                )
            st.plotly_chart(
                plot_mel_spectrogram(audio_array),
                use_container_width=True,
            )

    with tab2:
        result = st.session_state.analysis_results
        if result is None:
            st.info("Run an analysis first to see result-based visualizations.")
        else:
            imp = result.get("feature_importances", {})
            shap = result.get("shap_values", {})
            if imp:
                st.plotly_chart(plot_feature_importance(imp), use_container_width=True)
            if shap:
                st.plotly_chart(plot_shap_waterfall(shap), use_container_width=True)

    with tab3:
        st.markdown("**Demo visualizations with synthetic data**")
        demo_audio = np.random.randint(-20000, 20000, size=44100 * 3, dtype=np.int16)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_waveform(demo_audio, title="Demo Waveform"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_spectrogram(demo_audio, "Demo Spectrogram"), use_container_width=True)
        st.plotly_chart(plot_mel_spectrogram(demo_audio, "Demo Mel-Spectrogram"), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Training
# ---------------------------------------------------------------------------


def page_model_training() -> None:
    st.header("Model Training")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Training Configuration")
        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "xgboost", "svm", "neural_network", "logistic_regression"],
        )
        training_file = st.file_uploader(
            "Upload training data (CSV)",
            type=["csv"],
            key="train_upload",
        )

        if training_file is not None:
            df_preview = None
            try:
                import pandas as pd
                df_preview = pd.read_csv(training_file)
                st.dataframe(df_preview.head(10), use_container_width=True)
                st.caption(f"Rows: {len(df_preview)}, Columns: {len(df_preview.columns)}")
            except Exception as exc:
                st.warning(f"Could not preview: {exc}")

        params: dict[str, Any] = {}
        with st.expander("Hyperparameters"):
            if model_type in ("random_forest", "xgboost"):
                params["n_estimators"] = st.slider("n_estimators", 10, 500, 100)
                params["max_depth"] = st.slider("max_depth", 2, 50, 10)
            if model_type == "neural_network":
                params["hidden_layers"] = st.number_input("Hidden layers", 1, 5, 2)
                params["learning_rate"] = st.number_input("Learning rate", 0.0001, 0.1, 0.001, format="%.4f")
                params["epochs"] = st.number_input("Epochs", 5, 500, 50)

        if st.button("Start Training", type="primary", use_container_width=True):
            _start_training(model_type, training_file, params)

    with col2:
        st.subheader("Training Progress")
        job_id = st.session_state.training_job_id
        if job_id is None:
            st.info("No training job running.")
        else:
            _display_training_progress(job_id)


def _start_training(model_type: str, training_file: Any, params: dict) -> None:
    data: dict[str, Any] = {"model_type": model_type, "params": params}
    files = {}
    if training_file is not None:
        files["training_data"] = (
            training_file.name,
            training_file.getvalue(),
            "text/csv",
        )
    resp = _api_post("/training/start", data=data, files=files)
    if resp is not None:
        job = resp.json()
        st.session_state.training_job_id = job.get("job_id")
        st.success(f"Training started: {job.get('job_id')}")


def _display_training_progress(job_id: str) -> None:
    resp = _api_get(f"/training/status/{job_id}")
    if resp is None:
        return
    status = resp.json()
    state = status.get("status", "unknown")
    progress = status.get("progress", 0)

    st.metric("Status", state.upper())
    st.progress(progress / 100.0, text=f"{progress}%")

    metrics = status.get("metrics", {})
    if metrics:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Loss", f'{metrics.get("loss", 0):.4f}')
        mc2.metric("Accuracy", f'{metrics.get("accuracy", 0):.4f}')
        mc3.metric("F1", f'{metrics.get("f1", 0):.4f}')

    if st.button("Refresh"):
        st.rerun()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------


def page_about() -> None:
    st.header("About Voice Biomarker Platform v2")

    st.markdown("""
    ### Overview
    The Voice Biomarker Platform uses advanced machine learning to analyze vocal
    characteristics and identify potential health biomarkers from voice recordings.

    ### Technology Stack
    - **Frontend:** Streamlit
    - **Backend:** FastAPI
    - **ML Models:** scikit-learn, XGBoost, PyTorch
    - **Audio Processing:** librosa, scipy
    - **Explainability:** SHAP

    ### Features
    - Real-time audio analysis
    - Risk scoring with confidence intervals
    - SHAP-based model explainability
    - Multiple ML model support
    - Training pipeline with hyperparameter tuning
    """)

    st.divider()
    st.subheader("Medical Disclaimer")
    st.warning("""
    **This tool is for research and informational purposes only.**
    It is **NOT** a diagnostic device and should **NOT** be used as a
    substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for medical decisions.
    """)

    st.divider()
    st.subheader("Data Sources")
    st.markdown("""
    - Publicly available voice datasets
    - Clinical voice recordings (IRB-approved studies)
    - Synthetic data for augmentation
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    page = render_sidebar()

    dispatch = {
        "Upload & Analyze": page_upload_analyze,
        "Results History": page_results_history,
        "Visualizations": page_visualizations,
        "Model Training": page_model_training,
        "About": page_about,
    }
    dispatch[page]()


if __name__ == "__main__":
    main()
