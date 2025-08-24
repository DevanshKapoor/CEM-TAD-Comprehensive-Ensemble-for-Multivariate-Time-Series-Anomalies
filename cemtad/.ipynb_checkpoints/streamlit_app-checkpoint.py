from __future__ import annotations
import os, sys, io, tempfile
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import the pipeline no matter where Streamlit is launched from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_cemtad import run_pipeline

HONEYWELL_RED = "#E4002B"

# -------------------- caching + IO --------------------

@st.cache_data(show_spinner=False)
def read_head(path: str, nrows: int = 300) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

@st.cache_data(show_spinner=False)
def read_full(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_uploaded(uploaded) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uploaded.getbuffer()); tmp.flush(); tmp.close()
    return tmp.name

def is_scored(df_head: pd.DataFrame) -> bool:
    return "Abnormality_score" in df_head.columns

def autodetect_ts_col(df_head: pd.DataFrame) -> Optional[str]:
    for c in df_head.columns:
        if c.lower() == "time":
            return c
    best = None; best_ok = 0.0
    for c in df_head.columns:
        try:
            ok = pd.to_datetime(df_head[c], errors="coerce", infer_datetime_format=True).notna().mean()
            if ok > best_ok:
                best_ok, best = ok, c
        except Exception:
            pass
    return best

def suggest_normal_window(df: pd.DataFrame, ts_col: str) -> Tuple[str,str]:
    t = pd.to_datetime(df[ts_col], errors="coerce")
    tmin, tmax = t.min(), t.max()
    if pd.isna(tmin) or pd.isna(tmax):
        return "2004-01-01 00:00","2004-01-05 23:59"
    end = min(tmin + pd.Timedelta(hours=120), tmax)
    return tmin.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M")

# -------------------- analysis helpers --------------------

def segment_anomalies(scores: np.ndarray, thr: float, min_len: int = 1, gap: int = 0):
    """Return list of segments [ (start, end, peak_idx, peak_score) ] for scores >= thr."""
    segs = []
    start = None
    for i, s in enumerate(scores):
        if s >= thr and start is None:
            start = i
        elif (s < thr or i == len(scores)-1) and start is not None:
            end = i-1 if s < thr else i
            if end - start + 1 >= min_len:
                peak_idx = start + int(np.argmax(scores[start:end+1]))
                segs.append((start, end, peak_idx, float(scores[peak_idx])))
            start = None
    # optional merge small gaps
    if gap > 0 and segs:
        merged = [segs[0]]
        for s in segs[1:]:
            ps = merged[-1]
            if s[0] - ps[1] - 1 <= gap:
                new = (ps[0], s[1], ps[2] if ps[3] >= s[3] else s[2], max(ps[3], s[3]))
                merged[-1] = new
            else:
                merged.append(s)
        segs = merged
    return segs

def topfeatures_freq(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"top_feature_{i}" for i in range(1,8) if f"top_feature_{i}" in df.columns]
    s = pd.Series(dtype=int)
    for c in cols:
        s = s.add(df[c].dropna().astype(str).replace("", np.nan).dropna().value_counts(), fill_value=0)
    s = s.sort_values(ascending=False).astype(int)
    out = s.reset_index()
    out.columns = ["feature", "count"]   # pandas-compatible (avoids names= kw)
    return out

def numeric_feature_cols(df: pd.DataFrame, ts_col: str) -> List[str]:
    cols = [c for c in df.columns if c not in [ts_col, "Abnormality_score"] and not c.startswith("top_feature_")]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

def corr_delta_heatmap(df: pd.DataFrame, ts_col: str, train_mask: np.ndarray, win_slice: slice, max_feats: int = 30):
    feats = numeric_feature_cols(df, ts_col)
    if not feats:
        return None
    train_corr = df.loc[train_mask, feats].corr().fillna(0.0)
    event_corr = df.iloc[win_slice][feats].corr().fillna(0.0)
    delta = (event_corr - train_corr).abs()
    # focus on top rows/cols by total change
    rank = delta.sum(axis=1).sort_values(ascending=False).head(max_feats).index
    delta_small = delta.loc[rank, rank]
    fig = px.imshow(delta_small.values, x=rank, y=rank, color_continuous_scale="Reds", origin="lower")
    fig.update_layout(height=500, margin=dict(l=10,r=10,t=30,b=10), template="plotly_white",
                      title="Correlation Δ (event vs training)")
    return fig

# -------------------- visuals --------------------

def score_time_figure(df: pd.DataFrame, ts_col: str, normal_start: str, normal_end: str,
                      thr: float, segs: List[Tuple[int,int,int,float]]):
    t = pd.to_datetime(df[ts_col]); s = df["Abnormality_score"].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s, mode="lines", name="Score"))
    fig.add_hline(y=25, line_dash="dash", line_width=1)
    fig.add_hline(y=60, line_dash="dash", line_width=1)
    # training region
    ns, ne = pd.to_datetime(normal_start), pd.to_datetime(normal_end)
    fig.add_vrect(x0=ns, x1=ne, fillcolor="LightSalmon", opacity=0.25, line_width=0,
                  annotation_text="Training", annotation_position="top left")
    # event spans + peaks
    for (a,b,pk,pkv) in segs:
        fig.add_vrect(x0=t.iloc[a], x1=t.iloc[b], fillcolor="#E4002B", opacity=0.12, line_width=0)
        fig.add_trace(go.Scatter(x=[t.iloc[pk]], y=[pkv], mode="markers",
                                 marker=dict(size=8, color=HONEYWELL_RED),
                                 name="peak"))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=30,b=10),
                      yaxis_title="Score (0–100)", xaxis_title="Time",
                      legend=dict(orientation="h", y=1.1))
    return fig

def hist_train_rest(df: pd.DataFrame, ts_col: str, normal_start: str, normal_end: str):
    t = pd.to_datetime(df[ts_col]); mask = (t >= pd.to_datetime(normal_start)) & (t <= pd.to_datetime(normal_end))
    s_train, s_rest = df.loc[mask,"Abnormality_score"], df.loc[~mask,"Abnormality_score"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s_train, name="Train", opacity=0.75, nbinsx=60))
    fig.add_trace(go.Histogram(x=s_rest, name="Rest", opacity=0.6, nbinsx=60))
    fig.update_layout(barmode="overlay", template="plotly_white", height=320, margin=dict(l=10,r=10,t=30,b=10))
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Score (0–100)")
    return fig

def time_of_day_heatmap(df: pd.DataFrame, ts_col: str):
    t = pd.to_datetime(df[ts_col])
    tmp = pd.DataFrame({
        "dow": t.day_name(),
        "hour": t.dt.hour,
        "score": df["Abnormality_score"].astype(float)
    })
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    tmp["dow"] = pd.Categorical(tmp["dow"], categories=order, ordered=True)
    pivot = tmp.pivot_table(index="dow", columns="hour", values="score", aggfunc="mean")
    fig = px.imshow(pivot.values, x=list(pivot.columns), y=list(pivot.index),
                    color_continuous_scale="Reds", origin="lower", labels=dict(x="Hour", y="Day", color="Avg score"))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10,r=10,t=30,b=10),
                      title="Average score by day / hour")
    return fig

def feature_overlay(df: pd.DataFrame, ts_col: str, feats: List[str], center_idx: int, window: int = 120):
    start, end = max(0, center_idx - window), min(len(df), center_idx + window + 1)
    t = pd.to_datetime(df[ts_col].iloc[start:end])
    fig = go.Figure()
    for f in feats:
        if f in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df[f].iloc[start:end], mode="lines", name=f))
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Time")
    return fig

# -------------------- App --------------------

def app():
    st.set_page_config(page_title="CEM-TAD+ Explorer", layout="wide")

    st.markdown(
        f"""
        <div style="padding:16px 20px;border-radius:12px;background:{HONEYWELL_RED};color:#fff;margin-bottom:16px;">
            <div style="font-size:24px;font-weight:700;">CEM-TAD+ Anomaly Explorer</div>
            <div>Upload <b>raw</b> or <b>scored</b> CSV • GPU-ready • Explainable per-feature attributions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==== 1) INPUT CARD (main page, not sidebar) ====
    st.subheader("1) Upload & Configure")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload CSV (raw or scored)", type=["csv"])
        path_text = st.text_input("…or path to CSV (optional)", value="")
    with c2:
        mode = st.radio("Mode", ["Auto-detect (recommended)", "Use scored file", "Use raw file (run CEM-TAD+)"], index=0)

    csv_path = None; df_head = None
    if uploaded is not None:
        csv_path = save_uploaded(uploaded); df_head = read_head(csv_path)
    elif path_text and os.path.exists(path_text):
        csv_path = path_text; df_head = read_head(csv_path)
    else:
        st.info("Upload a CSV or paste a path to begin."); return

    ts_guess = autodetect_ts_col(df_head) or "Time"
    tsc = st.text_input("Timestamp column", value=ts_guess)

    try:
        df_peek = pd.read_csv(csv_path, usecols=[tsc])
        ns_sugg, ne_sugg = suggest_normal_window(df_peek, tsc)
    except Exception:
        ns_sugg, ne_sugg = "2004-01-01 00:00","2004-01-05 23:59"

    c3, c4 = st.columns(2)
    with c3: normal_start = st.text_input("Normal start (YYYY-MM-DD HH:MM)", value=ns_sugg)
    with c4: normal_end   = st.text_input("Normal end (YYYY-MM-DD HH:MM)",   value=ne_sugg)

    scored = is_scored(df_head)
    if mode == "Auto-detect (recommended)":
        mode = "Use scored file" if scored else "Use raw file (run CEM-TAD+)"

    with st.expander("Advanced (CEM-TAD+ parameters)", expanded=False):
        c5,c6,c7,c8 = st.columns(4)
        window = c5.slider("Window (minutes)", 20, 240, 60, 10)
        corr_window = c6.slider("Correlation window", 30, 360, 120, 10)
        patch = c7.slider("Patch length", 2, 24, 6, 1)
        epochs = c8.slider("Forecaster epochs", 5, 200, 30, 5)
        smoothing_alpha = st.slider("Smoothing α (EWMA)", 0.0, 1.0, 0.15, 0.01)

    run_clicked = st.button("▶ Run / Load", type="primary", use_container_width=True)
    if run_clicked:
    # do scoring/loading; set st.session_state.csv_path etc. as shown above

    # Proceed if we already have data in session
        csv_path = st.session_state.get("csv_path")
    if not csv_path:
        st.info("Upload a CSV or paste a path to begin."); st.stop()


    # ==== 2) PROCESS OR LOAD ====
    if mode == "Use raw file (run CEM-TAD+)" and not scored:
        with st.spinner("Scoring with CEM-TAD+…"):
            out_dir = tempfile.mkdtemp(prefix="cemtad_")
            out_csv = os.path.join(out_dir, "scored.csv")
            run_pipeline(
                input_csv=csv_path, output_csv=out_csv, timestamp_col=tsc,
                normal_start=normal_start, normal_end=normal_end,
                window=int(window), epochs=int(epochs), corr_window=int(corr_window), patch=int(patch),
                smoothing_alpha=float(smoothing_alpha), make_plots=False, plots_dir=out_dir
            )
            csv_path = out_csv

    df = read_full(csv_path)
    if "Abnormality_score" not in df.columns:
        st.error("CSV is not scored — ‘Abnormality_score’ column missing."); return

    # ==== 3) DASHBOARD ====
    st.subheader("2) Analysis Dashboard")

    t = pd.to_datetime(df[tsc])
    train_mask = (t >= pd.to_datetime(normal_start)) & (t <= pd.to_datetime(normal_end))
    s = df["Abnormality_score"].astype(float).values
    s_train = df.loc[train_mask, "Abnormality_score"].values
    s_rest  = df.loc[~train_mask, "Abnormality_score"].values


    # --- Suggested threshold ---
    mode_thr = st.radio(
        "Threshold suggestion mode",
        ["Youden-J (train vs rest)", "Target train FPR (0.5%)", "Legacy (max(60, p95 rest))"],
        index=0, horizontal=True
    )

    if mode_thr == "Youden-J (train vs rest)":
        from sklearn.metrics import roc_curve
        y = (~train_mask).astype(int).values
        fpr, tpr, th = roc_curve(y, df["Abnormality_score"].values)
        th_j = float(th[(tpr - fpr).argmax()])
        suggested_thr = max(40.0, th_j)
    elif mode_thr == "Target train FPR (0.5%)":
        s_train = df.loc[train_mask, "Abnormality_score"].values
        suggested_thr = float(np.quantile(s_train, 0.995)) if s_train.size else 60.0
    else:
        s_rest = df.loc[~train_mask, "Abnormality_score"].values
        suggested_thr = max(60.0, float(np.percentile(s_rest, 95))) if s_rest.size else 60.0


    if "threshold" not in st.session_state:
        st.session_state.threshold = int(round(suggested_thr))
    cthr1, cthr2 = st.columns([3,1])
    with cthr1:
        st.session_state.threshold = st.slider("Anomaly threshold", 0, 100, int(st.session_state.threshold), 1, key="thr_slider")
    with cthr2:
        if st.button(f"Use suggested ({int(round(suggested_thr))})"):
            st.session_state.threshold = int(round(suggested_thr))

    thr = float(st.session_state.threshold)
    segs = segment_anomalies(s, thr, min_len=1, gap=2)

    # KPIs
    mean_train = float(np.mean(s_train)) if len(s_train) else float("nan")
    max_train  = float(np.max(s_train)) if len(s_train) else float("nan")
    p99_train  = float(np.percentile(s_train, 99)) if len(s_train) else float("nan")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Features", str(len(numeric_feature_cols(df, tsc))))
    k3.metric("Train mean", f"{mean_train:.2f}")
    k4.metric("Train max", f"{max_train:.2f}")
    k5.metric("Train p99", f"{p99_train:.2f}")

    # TEXT INSIGHTS
    if len(segs) == 0:
        st.info(f"No points above threshold **{int(thr)}**. "
                f"Tip: try **{int(round(suggested_thr))}**, which is the 95th percentile of non-train scores.")
    else:
        n_points = int((s >= thr).sum())
        longest = max(segs, key=lambda z: z[1]-z[0]+1)
        st.success(f"Found **{len(segs)} anomaly events** (total **{n_points}** points ≥ {int(thr)}). "
                   f"Longest event spans **{longest[1]-longest[0]+1}** minutes; "
                   f"peak score **{longest[3]:.1f}** at row **{longest[2]}**.")

    tabs = st.tabs(["Overview", "Anomalies", "Contributors", "Distributions", "Correlation Δ"])

    # Overview
    with tabs[0]:
        st.plotly_chart(score_time_figure(df, tsc, normal_start, normal_end, thr, segs), use_container_width=True)
        # Download
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Download scored CSV", buf.getvalue(), "scored_output.csv", "text/csv")

        # Time-of-day/weekday heatmap
        st.plotly_chart(time_of_day_heatmap(df, tsc), use_container_width=True)

    # Anomalies (table + event viewer)
    with tabs[1]:
        anomalies = df[df["Abnormality_score"] >= thr].copy()
        st.dataframe(
            anomalies[[tsc, "Abnormality_score"] + [f"top_feature_{i}" for i in range(1,8) if f"top_feature_{i}" in df.columns]],
            use_container_width=True, height=320
        )
        if len(segs):
            # Event navigator
            options = {f"Event {i+1}: {a}–{b} (peak {pk:.1f} @ {pk_i})": (a,b,pk_i) for i,(a,b,pk_i,pk) in enumerate(segs)}
            label = list(options.keys())[0]
            label = st.selectbox("View event", options=list(options.keys()), index=0)
            a,b,pk_i = options[label]

            # Overlay features around the chosen event
            raw_cols = numeric_feature_cols(df, tsc)
            defaults = [df.iloc[pk_i].get(f"top_feature_{k}", "") for k in range(1,8)]
            defaults = [x for x in defaults if isinstance(x, str) and x in raw_cols][:3]
            feats = st.multiselect("Overlay raw features around this event", options=raw_cols, default=defaults)
            center = pk_i
            st.plotly_chart(feature_overlay(df, tsc, feats, center_idx=center, window=120), use_container_width=True)

    # Contributors
    with tabs[2]:
        freq = topfeatures_freq(df)
        fig = px.bar(freq.head(30), x="feature", y="count", template="plotly_white", height=360)
        fig.update_layout(title="Top features frequency (across Top-7 columns)")
        st.plotly_chart(fig, use_container_width=True)

        idx2 = st.number_input("Row index for Top-7", min_value=0, max_value=len(df)-1,
                               value=int(df["Abnormality_score"].idxmax()), step=1)
        tops = [df.iloc[idx2].get(f"top_feature_{i}", "") for i in range(1,8)]
        st.write("Top contributors:", ", ".join([x for x in tops if isinstance(x, str) and x != ""]) or "(none)")

    # Distributions
    with tabs[3]:
        st.plotly_chart(hist_train_rest(df, tsc, normal_start, normal_end), use_container_width=True)
        fig = px.scatter(df, x=pd.to_datetime(df[tsc]), y="Abnormality_score", opacity=0.6,
                         template="plotly_white", height=320)
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(xaxis_title="Time", yaxis_title="Score (0–100)")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Delta
    with tabs[4]:
        if len(segs):
            # choose event
            label_idx = st.selectbox("Choose event for correlation Δ", options=list(range(1, len(segs)+1)), index=0)
            a,b,pk_i,pk = segs[label_idx-1]
            # expand window around event for stability
            sl = slice(max(0, a-60), min(len(df), b+60))
            fig = corr_delta_heatmap(df, tsc, train_mask, sl, max_feats=30)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric features available to compute correlation deltas.")
        else:
            st.info("No events above the current threshold; lower it to see correlation deltas.")

if __name__ == "__main__":
    app()
