
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Display banner image
st.image("fusion_banner.png", use_column_width=True)

st.title("Fusion Mesh Simulation Tool")
st.write("Explore recursive feedback dynamics and quantum logic evolution with curvature-driven input.")

# User inputs
alpha = st.slider("α (alpha) - Memory strength", 0.0, 2.0, 1.0, 0.1)
beta = st.slider("β (beta) - Feedback weight", 0.0, 1.0, 0.8, 0.05)
timesteps = st.slider("Simulation Steps", 10, 200, 100, 10)
curvature_shape = st.selectbox("Curvature Signal Shape", ["sin", "cos", "spike", "flat"])

def generate_curvature(shape, timesteps):
    if shape == 'sin':
        return np.sin(np.linspace(0, 4 * np.pi, timesteps)) * 1e16
    elif shape == 'cos':
        return np.cos(np.linspace(0, 4 * np.pi, timesteps)) * 1e16
    elif shape == 'spike':
        data = np.zeros(timesteps)
        spike_indices = np.random.choice(range(timesteps), size=10, replace=False)
        data[spike_indices] = np.random.choice([-1, 1], size=10) * 2e16
        return data
    else:
        return np.zeros(timesteps)

g_series = generate_curvature(curvature_shape, timesteps)

x_t, y_t = 0, 1
F_t = 0
fusion_prob_series = []
feedback_series = []
logic_toggle_series = []

for t in range(timesteps):
    g_t = g_series[t]
    coherence = 1 - abs(x_t - y_t)
    fusion_prob_series.append(coherence)

    F_t = alpha * (x_t - y_t) + beta * F_t
    feedback_series.append(F_t)

    if g_t >= 0:
        x_t, y_t = 1, 0
    else:
        x_t, y_t = 0, 1

    logic_toggle_series.append((x_t, y_t))

data = pd.DataFrame({
    "Time Step": np.arange(timesteps),
    "Curvature Input g(t)": g_series,
    "Fusion Coherence P(x,y)": fusion_prob_series,
    "Recursive Feedback F(t)": feedback_series,
    "x_t": [x for x, y in logic_toggle_series],
    "y_t": [y for x, y in logic_toggle_series]
})

st.subheader("Simulation Results Plot")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data["Time Step"], data["Recursive Feedback F(t)"], label="F(t) - Feedback")
ax.plot(data["Time Step"], data["Fusion Coherence P(x,y)"], label="Coherence P(x,y)")
ax.plot(data["Time Step"], data["Curvature Input g(t)"] / 1e16, label="Curvature Input (scaled)")
ax.legend()
ax.set_xlabel("Time Step")
ax.set_title("Fusion Mesh Simulation – Adjustable Parameters")
ax.grid(True)
st.pyplot(fig)

st.subheader("Simulation Data")
st.dataframe(data)

csv = data.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="fusion_mesh_simulation_output.csv", mime="text/csv")
