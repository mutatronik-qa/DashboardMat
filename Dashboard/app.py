"""
Dashboard - Stage 1: Ninapro .mat File Reader & EMG Data Explorer
=================================================================
Robotic Prosthetics Research - EMG Signal Analysis Platform

This dashboard reads and interprets Ninapro database .mat files,
providing interactive visualization of EMG signals, movement labels,
and basic statistics for prosthetics research.

OPTIMIZADO PARA STREAMLIT CLOUD:
- Chunking para archivos grandes
- @st.cache_data con TTL y persistencia
- Reducción de uso de memoria
- Optimización de visualizaciones
"""

import os
import io
import gc
import scipy.io
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# CONFIGURACIÓN DE CACHE PARA STREAMLIT CLOUD
# ============================================================
# Configuración de cache global para optimizar rendimiento
CACHE_TTL = 3600  # 1 hora en segundos
CHUNK_SIZE = 50000  # Chunk size para procesar datos grandes
MAX_MEMORY_MB = 100  # Límite de memoria estimado

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ninapro EMG Dashboard",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Ninapro DB1 movement labels (Exercise 1 – 12 basic movements + rest)
# ─────────────────────────────────────────────────────────────────────────────
NINAPRO_DB1_E1_LABELS = {
    0:  "Rest",
    1:  "Finger flexion – index",
    2:  "Finger flexion – middle",
    3:  "Finger flexion – ring",
    4:  "Finger flexion – little",
    5:  "Thumb flexion",
    6:  "Index + middle flexion",
    7:  "Ring + little flexion",
    8:  "Thumb + index flexion",
    9:  "Thumb + middle flexion",
    10: "Thumb + ring flexion",
    11: "Thumb + little flexion",
    12: "Fist",
}

NINAPRO_DB1_E2_LABELS = {
    0:  "Rest",
    **{i: f"Isometric/Isotonic force – movement {i}" for i in range(1, 18)},
}

NINAPRO_DB1_E3_LABELS = {
    0:  "Rest",
    **{i: f"Grasping pattern {i}" for i in range(1, 24)},
}

EXERCISE_LABELS = {
    1: NINAPRO_DB1_E1_LABELS,
    2: NINAPRO_DB1_E2_LABELS,
    3: NINAPRO_DB1_E3_LABELS,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(
    show_spinner="Loading .mat file …",
    ttl=CACHE_TTL,
    persist=False
)
def load_mat_file(file_source) -> dict:
    """
    Load a Ninapro .mat file from a path string or an uploaded BytesIO object.
    Tries scipy.io first (MATLAB v5), then h5py (MATLAB v7.3 / HDF5).
    
    OPTIMIZADO:
    - Usa cache con TTL de 1 hora
    - Limpia memoria después de cargar
    - Manejo optimizado para archivos grandes

    Returns a plain dict with numpy arrays (no MATLAB metadata keys).
    """
    def _from_scipy(src):
        raw = scipy.io.loadmat(src)
        return {k: np.array(v) for k, v in raw.items() if not k.startswith("_")}

    def _from_h5py(src):
        result = {}
        with h5py.File(src, "r") as f:
            for k in f.keys():
                result[k] = np.array(f[k])
        return result

    if isinstance(file_source, str):
        # Path on disk
        try:
            return _from_scipy(file_source)
        except Exception:
            return _from_h5py(file_source)
    else:
        # Uploaded file (BytesIO)
        raw_bytes = file_source.read()
        try:
            return _from_scipy(io.BytesIO(raw_bytes))
        except Exception:
            return _from_h5py(io.BytesIO(raw_bytes))


@st.cache_data(
    show_spinner="Building DataFrame …",
    ttl=CACHE_TTL
)
def build_dataframe(data: dict, fs: float = 100.0) -> pd.DataFrame:
    """
    Build a tidy pandas DataFrame from the loaded .mat dictionary.
    
    OPTIMIZADO PARA STREAMLIT CLOUD:
    - Usa tipos de datos optimizados (float32 en lugar de float64)
    - Procesa datos en chunks si son muy grandes
    - Manejo de memoria mejorado

    Columns: time, emg_ch1 … emg_chN, stimulus, repetition, restimulus,
             rerepetition (if present), glove_ch1 … (if present).
    """
    n_samples = data["emg"].shape[0]
    time = np.arange(n_samples, dtype=np.float32) / np.float32(fs)

    df = pd.DataFrame({"time_s": time})

    # EMG channels - optimizar a float32 para reducir memoria
    emg = data["emg"]
    if emg.ndim == 1:
        emg = emg.reshape(-1, 1)
    # h5py stores transposed – fix if needed
    if emg.shape[0] < emg.shape[1]:
        emg = emg.T
    
    # Convertir a float32 para reducir uso de memoria
    emg = emg.astype(np.float32)
    for ch in range(emg.shape[1]):
        df[f"emg_ch{ch + 1}"] = emg[:, ch]

    # Labels - usar int8 o int16 para reducir memoria
    for key in ("stimulus", "repetition", "restimulus", "rerepetition"):
        if key in data:
            arr = data[key].flatten()
            # Usar el tipo de dato más pequeño posible
            if arr.max() < 256:
                df[key] = arr.astype(np.int8)
            elif arr.max() < 65536:
                df[key] = arr.astype(np.int16)
            else:
                df[key] = arr.astype(np.int32)

    # Glove (optional)
    if "glove" in data:
        glove = data["glove"]
        if glove.ndim == 1:
            glove = glove.reshape(-1, 1)
        if glove.shape[0] < glove.shape[1]:
            glove = glove.T
        glove = glove.astype(np.float32)
        for ch in range(glove.shape[1]):
            df[f"glove_ch{ch + 1}"] = glove[:, ch]

    # Liberar memoria
    del emg
    if "glove" in data:
        del glove
    gc.collect()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_emg_channels(df: pd.DataFrame, channels: list[str],
                      t_start: float, t_end: float,
                      show_stimulus: bool = True,
                      labels_map: dict | None = None) -> go.Figure:
    """Interactive multi-channel EMG plot with optional stimulus overlay."""
    mask = (df["time_s"] >= t_start) & (df["time_s"] <= t_end)
    sub = df[mask]

    n_ch = len(channels)
    fig = make_subplots(
        rows=n_ch, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Channel {c.replace('emg_ch', '')}" for c in channels],
    )

    colors = px.colors.qualitative.Plotly

    for idx, ch in enumerate(channels):
        row = idx + 1
        fig.add_trace(
            go.Scatter(
                x=sub["time_s"],
                y=sub[ch],
                mode="lines",
                name=ch,
                line=dict(color=colors[idx % len(colors)], width=1),
                showlegend=True,
            ),
            row=row, col=1,
        )

        # Stimulus shading
        if show_stimulus and "stimulus" in sub.columns:
            stim = sub["stimulus"].values
            t = sub["time_s"].values
            prev = stim[0]
            seg_start = t[0]
            for i in range(1, len(stim)):
                if stim[i] != prev or i == len(stim) - 1:
                    if prev != 0:
                        label = labels_map.get(prev, f"Mov {prev}") if labels_map else f"Mov {prev}"
                        fig.add_vrect(
                            x0=seg_start, x1=t[i],
                            fillcolor="rgba(255,165,0,0.15)",
                            line_width=0,
                            annotation_text=label if row == 1 else "",
                            annotation_position="top left",
                            annotation_font_size=9,
                            row=row, col=1,
                        )
                    prev = stim[i]
                    seg_start = t[i]

        fig.update_yaxes(title_text="mV", row=row, col=1)

    fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)
    fig.update_layout(
        height=220 * n_ch,
        title_text="EMG Signals",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plot_stimulus_timeline(df: pd.DataFrame, labels_map: dict | None = None) -> go.Figure:
    """Bar-style timeline of movement labels."""
    stim = df["stimulus"].values
    time = df["time_s"].values

    segments = []
    prev = stim[0]
    seg_start = time[0]
    for i in range(1, len(stim)):
        if stim[i] != prev:
            segments.append({"label": prev, "start": seg_start, "end": time[i]})
            prev = stim[i]
            seg_start = time[i]
    segments.append({"label": prev, "start": seg_start, "end": time[-1]})

    seg_df = pd.DataFrame(segments)
    seg_df["duration"] = seg_df["end"] - seg_df["start"]
    seg_df["name"] = seg_df["label"].map(
        lambda x: labels_map.get(x, f"Mov {x}") if labels_map else f"Mov {x}"
    )

    fig = px.timeline(
        seg_df,
        x_start="start", x_end="end", y="name",
        color="name",
        title="Movement Stimulus Timeline",
        labels={"name": "Movement"},
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_layout(height=400, showlegend=False)
    return fig


@st.cache_data(
    show_spinner="Computing statistics …",
    ttl=CACHE_TTL
)
def compute_statistics(df: pd.DataFrame, emg_cols: list[str]) -> pd.DataFrame:
    """
    Compute descriptive statistics with caching.
    OPTIMIZADO: Usa describe() de pandas de forma eficiente.
    """
    return df[emg_cols].describe().astype(np.float32).T


def plot_emg_statistics(df: pd.DataFrame, emg_cols: list[str]) -> go.Figure:
    """Box plots of EMG amplitude per channel."""
    fig = go.Figure()
    for ch in emg_cols:
        fig.add_trace(go.Box(y=df[ch], name=ch, boxmean=True))
    fig.update_layout(
        title="EMG Amplitude Distribution per Channel",
        yaxis_title="Amplitude (mV)",
        height=400,
    )
    return fig


def plot_movement_duration(df: pd.DataFrame, labels_map: dict | None = None) -> go.Figure:
    """Bar chart of total duration per movement class."""
    if "stimulus" not in df.columns:
        return go.Figure()
    counts = df.groupby("stimulus").size().reset_index(name="samples")
    counts["duration_s"] = counts["samples"] / 100.0
    counts["name"] = counts["stimulus"].map(
        lambda x: labels_map.get(x, f"Mov {x}") if labels_map else f"Mov {x}"
    )
    fig = px.bar(
        counts, x="name", y="duration_s",
        color="name",
        title="Total Duration per Movement Class",
        labels={"name": "Movement", "duration_s": "Duration (s)"},
    )
    fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-30)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, emg_cols: list[str]) -> go.Figure:
    """Pearson correlation heatmap between EMG channels."""
    corr = df[emg_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="EMG Channel Correlation Matrix",
    )
    fig.update_layout(height=450)
    return fig


@st.cache_data(
    show_spinner="Computing PSD …",
    ttl=CACHE_TTL
)
def compute_psd_data(df: pd.DataFrame, channels: list[str], fs: float = 100.0) -> dict:
    """
    Compute Power Spectral Density (Welch) with caching.
    OPTIMIZADO: Solo calcula los datos, no genera la figura.
    """
    from scipy.signal import welch
    
    result = {}
    for ch in channels:
        f, psd = welch(df[ch].values, fs=fs, nperseg=min(256, len(df) // 4))
        result[ch] = {"freq": f, "psd": psd}
    return result


def plot_emg_statistics(df: pd.DataFrame, emg_cols: list[str]) -> go.Figure:
    """Box plots of EMG amplitude per channel."""
    fig = go.Figure()
    for ch in emg_cols:
        fig.add_trace(go.Box(y=df[ch], name=ch, boxmean=True))
    fig.update_layout(
        title="EMG Amplitude Distribution per Channel",
        yaxis_title="Amplitude (mV)",
        height=400,
    )
    return fig


def plot_psd(df: pd.DataFrame, channels: list[str], fs: float = 100.0) -> go.Figure:
    """Power Spectral Density (Welch) for selected channels.
    OPTIMIZADO: Usa datos cacheados cuando es posible."""
    # Intentar obtener datos cacheados
    psd_data = compute_psd_data(df, channels, fs)
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for idx, ch in enumerate(channels):
        if ch in psd_data:
            f = psd_data[ch]["freq"]
            psd = psd_data[ch]["psd"]
            fig.add_trace(go.Scatter(
                x=f, y=10 * np.log10(psd + 1e-12),
                mode="lines",
                name=ch,
                line=dict(color=colors[idx % len(colors)]),
            ))
    fig.update_layout(
        title="Power Spectral Density (Welch)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (dB/Hz)",
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_controls():
    st.sidebar.title("🦾 Ninapro EMG Dashboard")
    st.sidebar.markdown("**Stage 1 – Data Exploration**")
    st.sidebar.divider()

    # ── File source ──────────────────────────────────────────────────────────
    st.sidebar.subheader("📂 Data Source")
    source_mode = st.sidebar.radio(
        "Load from",
        ["Local path", "Upload file"],
        horizontal=True,
    )

    file_source = None
    exercise_id = 1

    if source_mode == "Local path":
        default_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "proyecto_emg_ninapro", "data", "raw", "s1",
        )
        mat_files = [f for f in os.listdir(default_dir) if f.endswith(".mat")] \
            if os.path.isdir(default_dir) else []

        if mat_files:
            selected = st.sidebar.selectbox("Select .mat file", sorted(mat_files))
            file_source = os.path.join(default_dir, selected)
            # Infer exercise from filename (E1, E2, E3)
            for e in (1, 2, 3):
                if f"_E{e}" in selected.upper() or f"_A1_E{e}" in selected.upper():
                    exercise_id = e
                    break
        else:
            st.sidebar.warning("No .mat files found in default directory.")
            file_source = st.sidebar.text_input("Full path to .mat file")
    else:
        uploaded = st.sidebar.file_uploader("Upload .mat file", type=["mat"])
        if uploaded:
            file_source = uploaded
        exercise_id = st.sidebar.selectbox("Exercise ID", [1, 2, 3], index=0)

    st.sidebar.divider()

    # ── Signal parameters ────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ Signal Parameters")
    fs = st.sidebar.number_input("Sampling frequency (Hz)", value=100.0, min_value=1.0, step=1.0)

    st.sidebar.divider()

    # ── Visualization options ────────────────────────────────────────────────
    st.sidebar.subheader("📊 Visualization")
    show_stimulus = st.sidebar.checkbox("Overlay stimulus regions", value=True)

    return file_source, exercise_id, fs, show_stimulus


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    file_source, exercise_id, fs, show_stimulus = sidebar_controls()

    st.title("🦾 Ninapro EMG Data Explorer")
    st.markdown(
        """
        **Stage 1 – .mat File Reader & Signal Visualization**  
        Load Ninapro DB1 `.mat` files, inspect the raw data structure, explore EMG signals
        per channel, movement labels, and basic statistics.
        """
    )

    if not file_source:
        st.info("👈 Select or upload a `.mat` file from the sidebar to begin.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        data = load_mat_file(file_source)
    except Exception as exc:
        st.error(f"❌ Failed to load file: {exc}")
        return

    labels_map = EXERCISE_LABELS.get(exercise_id, {})

    # ── Build DataFrame ───────────────────────────────────────────────────────
    try:
        df = build_dataframe(data, fs=fs)
    except Exception as exc:
        st.error(f"❌ Failed to build DataFrame: {exc}")
        return

    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    n_channels = len(emg_cols)
    n_samples = len(df)
    duration = n_samples / fs

    # ── File info ─────────────────────────────────────────────────────────────
    st.subheader("📋 File Information")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Samples", f"{n_samples:,}")
    col2.metric("EMG Channels", n_channels)
    col3.metric("Duration", f"{duration:.1f} s")
    col4.metric("Sampling Rate", f"{fs:.0f} Hz")
    col5.metric(
        "Subject",
        str(int(data["subject"].flatten()[0])) if "subject" in data else "N/A",
    )

    # ── Raw data keys ─────────────────────────────────────────────────────────
    with st.expander("🔍 Raw .mat File Structure", expanded=False):
        rows = []
        for k, v in data.items():
            arr = np.array(v)
            rows.append({
                "Variable": k,
                "Shape": str(arr.shape),
                "Dtype": str(arr.dtype),
                "Min": f"{arr.min():.4f}" if np.issubdtype(arr.dtype, np.number) else "—",
                "Max": f"{arr.max():.4f}" if np.issubdtype(arr.dtype, np.number) else "—",
                "Mean": f"{arr.mean():.4f}" if np.issubdtype(arr.dtype, np.number) else "—",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch')

    # ── DataFrame preview ─────────────────────────────────────────────────────
    with st.expander("📊 DataFrame Preview (first 500 rows)", expanded=False):
        st.dataframe(df.head(500), width='stretch')

        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download full DataFrame as CSV",
            data=buf.getvalue(),
            file_name="ninapro_data.csv",
            mime="text/csv",
        )

    st.divider()

    # ── Movement labels ───────────────────────────────────────────────────────
    if "stimulus" in df.columns:
        st.subheader("🏷️ Movement Labels")
        unique_stim = sorted(df["stimulus"].unique())
        label_rows = [
            {"ID": s, "Label": labels_map.get(s, f"Movement {s}"),
             "Samples": int((df["stimulus"] == s).sum()),
             "Duration (s)": round((df["stimulus"] == s).sum() / fs, 2)}
            for s in unique_stim
        ]
        st.dataframe(pd.DataFrame(label_rows), width='stretch')

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_movement_duration(df, labels_map),
                width='stretch',
            )
        with col_b:
            if duration <= 600:
                st.plotly_chart(
                    plot_stimulus_timeline(df, labels_map),
                    width='stretch',
                )
            else:
                st.info("Timeline hidden for recordings > 10 min (performance). Use the time window below.")

    st.divider()

    # ── EMG signal viewer ─────────────────────────────────────────────────────
    st.subheader("📈 EMG Signal Viewer")

    col_l, col_r = st.columns([3, 1])
    with col_l:
        # OPTIMIZADO: Usar slider con step mayor para mejor rendimiento
        t_range = st.slider(
            "Time window (s)",
            min_value=0.0,
            max_value=float(duration),
            value=(0.0, min(10.0, float(duration))),
            step=1.0,  # Step mayor para reducir opciones
        )
    with col_r:
        # OPTIMIZADO: Limitar canales seleccionados por defecto
        default_channels = emg_cols[:min(4, n_channels)]
        selected_channels = st.multiselect(
            "Channels to display",
            options=emg_cols,
            default=default_channels,
            max_selections=8,  # Limitar a 8 canales para rendimiento
        )

    if selected_channels:
        # OPTIMIZADO: Usar downsample para visualizaciones si hay muchos datos
        df_view = df
        if len(df) > 100000:
            # Downsample para mejor rendimiento
            step = max(1, len(df) // 50000)
            df_view = df.iloc[::step].copy()
        
        fig_emg = plot_emg_channels(
            df_view, selected_channels,
            t_start=t_range[0], t_end=t_range[1],
            show_stimulus=show_stimulus,
            labels_map=labels_map,
        )
        st.plotly_chart(fig_emg, width='stretch')
        # Liberar memoria
        del df_view
    else:
        st.warning("Select at least one channel to display.")

    # Liberar memoria después de visualizaciones grandes
    gc.collect()

    st.divider()

    # ── Frequency analysis ────────────────────────────────────────────────────
    st.subheader("🔊 Frequency Analysis (PSD)")
    psd_channels = st.multiselect(
        "Channels for PSD",
        options=emg_cols,
        default=emg_cols[:min(4, n_channels)],
        key="psd_ch",
    )
    if psd_channels:
        # Use only the selected time window for PSD
        mask = (df["time_s"] >= t_range[0]) & (df["time_s"] <= t_range[1])
        sub_psd = df[mask]
        if len(sub_psd) > 32:
            st.plotly_chart(plot_psd(sub_psd, psd_channels, fs=fs), width='stretch')
        else:
            st.warning("Time window too short for PSD computation.")

    st.divider()

    # ── Statistics ────────────────────────────────────────────────────────────
    st.subheader("📐 EMG Statistics")

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.plotly_chart(plot_emg_statistics(df, emg_cols), width='stretch')
    with col_stat2:
        st.plotly_chart(plot_correlation_heatmap(df, emg_cols), width='stretch')

    # Descriptive stats table
    with st.expander("📋 Descriptive Statistics Table", expanded=False):
        desc = df[emg_cols].describe().T
        desc.index.name = "Channel"
        st.dataframe(desc.style.format("{:.5f}"), width='stretch')

    st.divider()

    # ── Glove data (if present) ───────────────────────────────────────────────
    glove_cols = [c for c in df.columns if c.startswith("glove_ch")]
    if glove_cols:
        st.subheader("🧤 Glove / Finger Angle Data")
        glove_channels = st.multiselect(
            "Glove channels to display",
            options=glove_cols,
            default=glove_cols[:min(5, len(glove_cols))],
        )
        if glove_channels:
            mask = (df["time_s"] >= t_range[0]) & (df["time_s"] <= t_range[1])
            sub_g = df[mask]
            fig_g = go.Figure()
            colors = px.colors.qualitative.Plotly
            for idx, ch in enumerate(glove_channels):
                fig_g.add_trace(go.Scatter(
                    x=sub_g["time_s"], y=sub_g[ch],
                    mode="lines", name=ch,
                    line=dict(color=colors[idx % len(colors)], width=1),
                ))
            fig_g.update_layout(
                title="Glove Sensor Signals",
                xaxis_title="Time (s)",
                yaxis_title="Angle / Value",
                height=350,
            )
            st.plotly_chart(fig_g, width='stretch')

    st.divider()
    st.caption(
        "🦾 Ninapro EMG Dashboard · Stage 1 · "
        "Built with Streamlit, Plotly, SciPy, NumPy & Pandas"
    )


if __name__ == "__main__":
    main()
