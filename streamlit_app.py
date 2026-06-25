import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Takens' Theorem Explorer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark scholarly palette */
    :root {
        --bg: #0f1117;
        --surface: #1a1d27;
        --accent: #7c6af7;
        --accent2: #f7c06a;
        --text: #e8eaf0;
        --muted: #8b8fa8;
        --border: #2a2d3e;
    }
    .stApp { background: var(--bg); color: var(--text); }
    section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }

    .hero-banner {
        background: linear-gradient(135deg, #1a1d27 0%, #12152b 50%, #1a1d27 100%);
        border: 1px solid #2a2d3e;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 40%, rgba(124,106,247,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 60%, rgba(247,192,106,0.06) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-title { font-size: 2.1rem; font-weight: 700; color: #e8eaf0; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
    .hero-sub   { font-size: 1rem; color: #8b8fa8; margin: 0; }

    .theorem-box {
        background: linear-gradient(135deg, rgba(124,106,247,0.08), rgba(124,106,247,0.03));
        border: 1px solid rgba(124,106,247,0.35);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .theorem-box h4 { color: #a99cf9; margin: 0 0 0.5rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .theorem-box p  { color: #c8cbdb; margin: 0; font-size: 0.93rem; line-height: 1.6; }

    .metric-row { display: flex; gap: 1rem; margin: 0.8rem 0; }
    .metric-card {
        flex: 1;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .metric-card .val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .metric-card .lbl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

    .step-badge {
        display: inline-block;
        background: rgba(124,106,247,0.18);
        color: #a99cf9;
        border: 1px solid rgba(124,106,247,0.4);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .info-pill {
        display: inline-block;
        background: rgba(247,192,106,0.12);
        color: #f7c06a;
        border: 1px solid rgba(247,192,106,0.3);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.8rem;
        margin: 2px 3px;
    }
    div[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
    .stSlider > div > div > div { background: var(--accent) !important; }
    .stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border) !important; }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── ODE Systems ────────────────────────────────────────────────────────────────
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

def rossler(t, state, a=0.1, b=0.1, c=14):
    x, y, z = state
    return [-y - z, x + a*y, b + z*(x - c)]

def duffing(t, state, alpha=1, beta=-1, delta=0.2, gamma=0.3, omega=1.2):
    x, v = state
    return [v, -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]

def vanderpol(t, state, mu=2.0):
    x, v = state
    return [v, mu*(1 - x**2)*v - x]

SYSTEMS = {
    "Lorenz": {
        "fn": lorenz, "dim": 3, "ic": [1.0, 1.0, 1.0],
        "t_span": (0, 60), "dt": 0.01,
        "desc": "The canonical chaotic butterfly attractor (σ=10, ρ=28, β=8/3).",
        "obs_idx": 0, "obs_name": "x(t)"
    },
    "Rössler": {
        "fn": rossler, "dim": 3, "ic": [0.0, 1.0, 0.0],
        "t_span": (0, 150), "dt": 0.02,
        "desc": "A simpler chaotic attractor with one folded band (a=0.1, b=0.1, c=14).",
        "obs_idx": 0, "obs_name": "x(t)"
    },
    "Duffing": {
        "fn": duffing, "dim": 2, "ic": [0.5, 0.0],
        "t_span": (0, 200), "dt": 0.02,
        "desc": "Periodically driven nonlinear oscillator; chaotic for these parameters.",
        "obs_idx": 0, "obs_name": "x(t)"
    },
    "Van der Pol": {
        "fn": vanderpol, "dim": 2, "ic": [2.0, 0.0],
        "t_span": (0, 100), "dt": 0.02,
        "desc": "Relaxation oscillator with a stable limit cycle (μ=2).",
        "obs_idx": 0, "obs_name": "x(t)"
    },
}

@st.cache_data(show_spinner=False)
def integrate_system(name):
    cfg = SYSTEMS[name]
    sol = solve_ivp(cfg["fn"], cfg["t_span"], cfg["ic"],
                    max_step=cfg["dt"], dense_output=False, method="RK45")
    t = sol.t
    y = sol.y
    # discard transient (first 20%)
    cut = int(0.20 * len(t))
    return t[cut:], y[:, cut:]


# ── Delay embedding ────────────────────────────────────────────────────────────
def delay_embed(ts, tau, dim):
    N = len(ts) - (dim - 1) * tau
    if N <= 0:
        return None
    out = np.zeros((N, dim))
    for d in range(dim):
        out[:, d] = ts[d * tau: d * tau + N]
    return out


def auto_mutual_info(ts, max_lag=100):
    """Estimate first minimum of average mutual information via histogram."""
    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-12)
    n_bins = int(np.sqrt(len(ts) / 5)) + 1
    ami = []
    for lag in range(1, max_lag + 1):
        x = ts_norm[:-lag]
        y = ts_norm[lag:]
        hist2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = hist2d / hist2d.sum()
        px  = pxy.sum(axis=1, keepdims=True)
        py  = pxy.sum(axis=0, keepdims=True)
        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px * py + 1e-12)[mask]))
        ami.append(mi)
    return np.array(ami)


def false_nearest_neighbors(ts, tau, max_dim=8, threshold=10.0):
    """Compute FNN fraction for each embedding dimension."""
    fnn = []
    for d in range(1, max_dim + 1):
        pts = delay_embed(ts, tau, d)
        pts_next = delay_embed(ts, tau, d + 1)
        if pts is None or pts_next is None:
            fnn.append(0.0)
            continue
        N = min(len(pts), len(pts_next), 500)   # subsample for speed
        idx = np.random.choice(min(len(pts), len(pts_next)), N, replace=False)
        pts_s = pts[idx]
        pts_next_s = pts_next[idx]

        from scipy.spatial import cKDTree
        tree = cKDTree(pts_s)
        dists, neighbors = tree.query(pts_s, k=2)
        r_d = dists[:, 1] + 1e-12
        r_d1 = np.linalg.norm(pts_next_s - pts_next_s[neighbors[:, 1]], axis=1)
        ratio = r_d1 / r_d
        fnn.append(np.mean(ratio > threshold))
    return np.array(fnn)


# ── Plotting helpers ────────────────────────────────────────────────────────────
PALETTE = ["#7c6af7", "#f7c06a", "#5ecfc0", "#f76a8a", "#a3e4d7"]

def plot_time_series(t, obs, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=obs, mode="lines",
                             line=dict(color=PALETTE[0], width=1.2), name=name))
    fig.update_layout(
        title=dict(text=f"Observed scalar time series  —  {name}", font=dict(color="#c8cbdb", size=14)),
        xaxis=dict(title="Time", gridcolor="#2a2d3e", color="#8b8fa8"),
        yaxis=dict(title=name, gridcolor="#2a2d3e", color="#8b8fa8"),
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#8b8fa8"), margin=dict(l=50, r=20, t=50, b=40),
        height=260
    )
    return fig


def plot_true_attractor(y, name, dim3):
    if dim3:
        fig = go.Figure(go.Scatter3d(
            x=y[0], y=y[1], z=y[2],
            mode="lines",
            line=dict(color=PALETTE[0], width=1.5,
                      colorscale="Plasma",
                      color=np.linspace(0, 1, len(y[0]))),
        ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="x", backgroundcolor="#0f1117", gridcolor="#2a2d3e", color="#8b8fa8"),
                yaxis=dict(title="y", backgroundcolor="#0f1117", gridcolor="#2a2d3e", color="#8b8fa8"),
                zaxis=dict(title="z", backgroundcolor="#0f1117", gridcolor="#2a2d3e", color="#8b8fa8"),
                bgcolor="#0f1117",
            ),
        )
    else:
        fig = go.Figure(go.Scatter(
            x=y[0], y=y[1], mode="lines",
            line=dict(color=PALETTE[0], width=1)
        ))
        fig.update_layout(
            xaxis=dict(title="x", gridcolor="#2a2d3e", color="#8b8fa8"),
            yaxis=dict(title="ẋ", gridcolor="#2a2d3e", color="#8b8fa8"),
            plot_bgcolor="#0f1117",
        )
    fig.update_layout(
        title=dict(text=f"True attractor  —  {name}", font=dict(color="#c8cbdb", size=14)),
        paper_bgcolor="#0f1117", font=dict(color="#8b8fa8"),
        margin=dict(l=10, r=10, t=50, b=10), height=420
    )
    return fig


def plot_reconstruction(emb, dim):
    if dim == 2:
        fig = go.Figure(go.Scatter(
            x=emb[:, 0], y=emb[:, 1], mode="lines",
            line=dict(color=PALETTE[1], width=1,
                      colorscale="Viridis",
                      color=np.linspace(0, 1, len(emb[:, 0]))),
        ))
        fig.update_layout(
            xaxis=dict(title="s(t)", gridcolor="#2a2d3e", color="#8b8fa8"),
            yaxis=dict(title="s(t+τ)", gridcolor="#2a2d3e", color="#8b8fa8"),
            plot_bgcolor="#0f1117",
        )
    else:
        fig = go.Figure(go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="lines",
            line=dict(color=PALETTE[1], width=1.5,
                      colorscale="Viridis",
                      color=np.linspace(0, 1, len(emb[:, 0]))),
        ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="s(t)", backgroundcolor="#0f1117", gridcolor="#2a2d3e", color="#8b8fa8"),
                yaxis=dict(title="s(t+τ)", backgroundcolor="#0f1117", gridcolor="#2a2d3e", color="#8b8fa8"),
                zaxis=dict(title="s(t+2τ)", backgroundcolor="#0f1117", gridcolor="#2a2d3e", color="#8b8fa8"),
                bgcolor="#0f1117",
            ),
        )
    fig.update_layout(
        title=dict(text=f"Reconstructed attractor  (m={dim}, τ={emb.shape[0]})", font=dict(color="#c8cbdb", size=14)),
        paper_bgcolor="#0f1117", font=dict(color="#8b8fa8"),
        margin=dict(l=10, r=10, t=50, b=10), height=420
    )
    return fig


def plot_ami(ami, tau_opt):
    lags = np.arange(1, len(ami) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=ami, mode="lines+markers",
                             line=dict(color=PALETTE[2], width=1.8),
                             marker=dict(size=4, color=PALETTE[2]), name="AMI"))
    fig.add_vline(x=tau_opt, line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f"τ = {tau_opt}", annotation_font_color=PALETTE[3])
    fig.update_layout(
        title=dict(text="Average Mutual Information  →  optimal τ", font=dict(color="#c8cbdb", size=13)),
        xaxis=dict(title="Lag", gridcolor="#2a2d3e", color="#8b8fa8"),
        yaxis=dict(title="AMI", gridcolor="#2a2d3e", color="#8b8fa8"),
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#8b8fa8"), margin=dict(l=50, r=20, t=50, b=40), height=280
    )
    return fig


def plot_fnn(fnn, dim_opt):
    dims = np.arange(1, len(fnn) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dims, y=fnn * 100, marker_color=PALETTE[4], name="FNN %"))
    fig.add_vline(x=dim_opt, line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f"m = {dim_opt}", annotation_font_color=PALETTE[3])
    fig.update_layout(
        title=dict(text="False Nearest Neighbours  →  optimal m", font=dict(color="#c8cbdb", size=13)),
        xaxis=dict(title="Embedding dim m", gridcolor="#2a2d3e", color="#8b8fa8", tickvals=list(dims)),
        yaxis=dict(title="FNN (%)", gridcolor="#2a2d3e", color="#8b8fa8"),
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#8b8fa8"), margin=dict(l=50, r=20, t=50, b=40), height=280
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌀 System")
    system_name = st.selectbox("Dynamical system", list(SYSTEMS.keys()), index=0)
    cfg = SYSTEMS[system_name]
    st.caption(cfg["desc"])
    st.markdown("---")

    st.markdown("### ⚙️ Embedding parameters")
    auto_params = st.toggle("Auto-select τ and m", value=True)

    max_lag = st.slider("AMI max lag (for τ search)", 20, 200, 80, 5,
                        help="Upper bound when scanning for the first AMI minimum.")

    if not auto_params:
        tau_manual = st.slider("Time delay  τ", 1, 100, 10)
        dim_manual = st.slider("Embedding dimension  m", 2, 6, 3)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
**Takens' embedding theorem** (1981) states that, given a smooth dynamical system and a *generic* scalar observation function, the delay-embedding map

> **Φ**: x(t) ↦ (s(t), s(t+τ), …, s(t+(m-1)τ))

is a **diffeomorphism** onto the attractor when m ≥ 2d+1 (d = attractor dimension).

This lets you reconstruct the full attractor topology from a *single* time series.
""")

# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🌀 Takens' Theorem Explorer</div>
  <div class="hero-sub">Reconstruct strange attractors from a single scalar observable using delay embeddings</div>
</div>
""", unsafe_allow_html=True)

# Theorem box
st.markdown("""
<div class="theorem-box">
<h4>Takens' Embedding Theorem  (1981)</h4>
<p>
For a compact smooth manifold M of dimension d and a generic smooth vector field and observation function,
the delay embedding map <b>Φ<sub>τ,m</sub> : M → ℝ<sup>m</sup></b> — mapping each state to 
(s(t), s(t+τ), …, s(t+(m−1)τ)) — is an <b>embedding</b> whenever m ≥ 2d + 1.
The reconstructed attractor is <em>diffeomorphic</em> to the original; all topological invariants 
(Lyapunov exponents, fractal dimension, …) are preserved.
</p>
</div>
""", unsafe_allow_html=True)

# ── Integrate ─────────────────────────────────────────────────────────────────
with st.spinner("Integrating the system…"):
    t, y = integrate_system(system_name)

obs = y[cfg["obs_idx"]]
obs_name = cfg["obs_name"]

# ── Step 1 : time series ───────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 — Observe</div>', unsafe_allow_html=True)
st.markdown("We measure only **one scalar** from the system — exactly as in real experiments.")
st.plotly_chart(plot_time_series(t, obs, obs_name), use_container_width=True)

# ── Step 2 : parameter selection ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-badge">Step 2 — Choose τ and m</div>', unsafe_allow_html=True)

col_ami, col_fnn = st.columns(2)

with st.spinner("Computing AMI and FNN…"):
    ami = auto_mutual_info(obs, max_lag=max_lag)
    peaks_neg, _ = find_peaks(-ami)
    tau_auto = int(peaks_neg[0]) + 1 if len(peaks_neg) > 0 else max(1, max_lag // 5)

    fnn = false_nearest_neighbors(obs, tau_auto, max_dim=7)
    dim_auto = int(np.argmax(fnn < 0.05) + 1) if np.any(fnn < 0.05) else int(np.argmin(fnn) + 1)
    dim_auto = max(2, min(dim_auto, 3))

tau = tau_auto if auto_params else tau_manual
dim = dim_auto if auto_params else dim_manual

with col_ami:
    st.markdown("**Average Mutual Information** finds the first minimum — the lag at which s(t+τ) carries *new* information about s(t).")
    st.plotly_chart(plot_ami(ami, tau), use_container_width=True)

with col_fnn:
    st.markdown("**False Nearest Neighbours** estimates the minimum embedding dimension: when FNN → 0, unfolding is complete.")
    st.plotly_chart(plot_fnn(fnn, dim), use_container_width=True)

# Metrics
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="val">{tau}</div><div class="lbl">Optimal delay  τ</div></div>
  <div class="metric-card"><div class="val">{dim}</div><div class="lbl">Embedding dim  m</div></div>
  <div class="metric-card"><div class="val">{len(obs):,}</div><div class="lbl">Time-series points</div></div>
  <div class="metric-card"><div class="val">{len(obs) - (dim-1)*tau:,}</div><div class="lbl">Embedded points</div></div>
</div>
""", unsafe_allow_html=True)

# ── Step 3 : side-by-side attractors ──────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-badge">Step 3 — Reconstruct</div>', unsafe_allow_html=True)
st.markdown("The delay-embedded attractor (right) is **diffeomorphic** to the true one (left) — same shape, computed from a single time series.")

emb = delay_embed(obs, tau, dim)

col_true, col_rec = st.columns(2)

with col_true:
    is_3d = cfg["dim"] == 3
    st.plotly_chart(plot_true_attractor(y, system_name, is_3d), use_container_width=True)

with col_rec:
    if emb is not None:
        plot_dim = min(dim, 3)
        st.plotly_chart(plot_reconstruction(emb, plot_dim), use_container_width=True)
    else:
        st.error("Embedding failed — try a smaller τ or m.")

# ── Step 4 : interactive τ sweep ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-badge">Step 4 — Explore</div>', unsafe_allow_html=True)
st.markdown("Drag the slider to see how the reconstructed attractor degrades when τ is poorly chosen.")

tau_sweep = st.slider("Sweep τ to see its effect", 1, min(120, max_lag), tau, 1)
emb_sweep = delay_embed(obs, tau_sweep, min(dim, 3))

col_info, col_sweep = st.columns([1, 2])
with col_info:
    quality = "✅ Good" if abs(tau_sweep - tau_auto) <= tau_auto // 2 else ("⚠️ Too small — correlated" if tau_sweep < tau_auto // 2 else "⚠️ Too large — folded")
    st.markdown(f"""
<div class="theorem-box" style="margin-top:3.5rem">
<h4>Embedding quality</h4>
<p>
<span class="info-pill">τ chosen = {tau_sweep}</span>
<span class="info-pill">τ optimal = {tau_auto}</span>
<br><br>
<b>{quality}</b><br><br>
Small τ → consecutive delays are <em>too correlated</em>, the trajectory collapses onto the diagonal.<br><br>
Large τ → the trajectory folds and self-intersects, destroying the diffeomorphism.
</p>
</div>
""", unsafe_allow_html=True)
with col_sweep:
    if emb_sweep is not None:
        st.plotly_chart(plot_reconstruction(emb_sweep, min(dim, 3)), use_container_width=True)

# ── Explainer ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Mathematical background", expanded=False):
    st.markdown("""
### Takens' Theorem — key ideas

**Setup.** Let ϕ_t be a smooth flow on a compact manifold M ⊂ ℝ^d, and let h : M → ℝ be a smooth observation function.  
Define the delay map

> **Φ_{h,ϕ}(x) = (h(x), h(ϕ_τ(x)), …, h(ϕ_{(m-1)τ}(x)))**

**Theorem (Takens, 1981).** For generic pairs (ϕ, h) and m ≥ 2d + 1, the map Φ_{h,ϕ} is an *embedding* — i.e., an injective immersion.

**Consequences**
- The image of Φ is diffeomorphic to M.
- All smooth invariants are preserved: Lyapunov spectrum, correlation dimension, topological entropy.
- Causally equivalent systems have the same reconstructed attractor up to diffeomorphism.

**Choosing τ**  
τ should be large enough that s(t) and s(t+τ) are *statistically independent* but small enough that the orbit hasn't folded.  
→ **First minimum of Average Mutual Information** (Fraser & Swinney, 1986).

**Choosing m**  
m must be large enough to *unfold* the attractor so that no two distinct states project to the same point.  
→ **False Nearest Neighbours** method (Kennel, Brown & Abarbanel, 1992).

**Generalisations**  
- Sauer, Yorke & Casdagli (1991) extended to fractal attractors: m > 2 d_box suffices.  
- Aeyels, Deyle & Sugihara extended to noisy and multivariate settings.
""")

st.markdown("""
<div style='text-align:center; color:#3a3d50; font-size:0.78rem; margin-top:2rem;'>
Built with Streamlit · Takens (1981), Fraser & Swinney (1986), Kennel et al. (1992)
</div>
""", unsafe_allow_html=True)