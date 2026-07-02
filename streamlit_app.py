import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint

# --- Configuration ---
st.set_page_config(page_title="Takens' Theorem Explorer", layout="wide")

# --- System Equations ---
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# --- Helper Functions ---
@st.cache_data
def generate_data(sigma, rho, beta, t_max, dt):
    """Simulate the Lorenz system."""
    t = np.arange(0, t_max, dt)
    initial_state = [1.0, 1.0, 1.0]
    states = odeint(lorenz, initial_state, t, args=(sigma, rho, beta))
    return t, states

def time_delay_embedding(time_series, delay, dimension):
    """Reconstruct phase space using time delay embedding."""
    N = len(time_series)
    embedded = np.zeros((N - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        embedded[:, i] = time_series[i * delay : N - (dimension - 1 - i) * delay]
    return embedded

# --- UI Setup ---
st.title("Takens' Theorem & Attractor Reconstruction")
st.markdown("""
Takens' theorem states that we can reconstruct a chaotic system's multidimensional dynamics from a single time series. 
Here, we reconstruct the geometry of the Lorenz attractor using only delayed versions of $x$.
""")

# Sidebar inputs
with st.sidebar:
    st.header("1. Lorenz Parameters")
    sigma = st.slider("Sigma (σ)", 0.0, 20.0, 10.0, 0.5)
    rho = st.slider("Rho (ρ)", 0.0, 50.0, 28.0, 0.5)
    beta = st.slider("Beta (β)", 0.0, 10.0, 8.0/3.0, 0.1)
    
    st.header("2. Simulation Settings")
    t_max = st.number_input("Time duration", 10, 100, 40)
    dt = st.number_input("Time step (dt)", 0.001, 0.1, 0.01, format="%.3f")
    
    st.header("3. Takens' Parameters")
    st.markdown("Adjust the delay (τ) to see how it affects the reconstructed geometry.")
    delay_index = st.slider("Time Delay (τ) index", 1, 100, 15)

# --- Data Generation ---
t, states = generate_data(sigma, rho, beta, t_max, dt)
x, y, z = states[:, 0], states[:, 1], states[:, 2]

# Perform embedding using ONLY the x-variable
# We use an embedding dimension of 3 to match the original geometry's visual appeal
embedded_states = time_delay_embedding(x, delay_index, dimension=3)
x_recon = embedded_states[:, 0]
y_recon = embedded_states[:, 1]
z_recon = embedded_states[:, 2]

st.subheader("Original 3D Attractor")
st.markdown("The true phase space using $x$, $y$, and $z$.")
fig_orig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(color=t, colorscale='Viridis', width=2)
)])
fig_orig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
st.plotly_chart(fig_orig, use_container_width=True)

# Plot the 1D time series of X
st.subheader("$x(t)$ time series")
fig_ts = go.Figure(data=[go.Scatter(x=t, y=x, mode='lines', line=dict(color='#ff7f0e'))])
fig_ts.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=250, xaxis_title="Time", yaxis_title="x")
st.plotly_chart(fig_ts, use_container_width=True)

st.subheader("Reconstructed 3D Attractor")
st.markdown(f"Reconstructed using $x(t)$, $x(t - \\tau)$, and $x(t - 2\\tau)$.")
# Time vector adjusted for the delay truncation for coloring
t_recon = t[:len(x_recon)]

fig_recon = go.Figure(data=[go.Scatter3d(
    x=x_recon, y=y_recon, z=z_recon,
    mode='lines',
    line=dict(color=t_recon, colorscale='Plasma', width=2)
)])
fig_recon.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
st.plotly_chart(fig_recon, use_container_width=True)

