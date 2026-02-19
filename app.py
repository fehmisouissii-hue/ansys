import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import torch
import torch.nn as nn
import tempfile

# 1. The AI Agent Logic (PINN-style)
class StressAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 1) # Predicts 1 stress value per point
        )
    def forward(self, x): return self.net(x)

# 2. The Website Layout
st.set_page_config(page_title="AI Stress Lab", layout="wide")
st.title("🦷 AI Hip Implant Analysis")

uploaded_file = st.file_uploader("Upload Hip Implant STL File", type=['stl'])
load = st.slider("Applied Load (Newtons)", 100, 5000, 1000)

if uploaded_file:
    # Save upload to a temporary file so PyVista can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    if st.button("RUN AI SIMULATION"):
        # Load the 3D Model
        mesh = pv.read(path)
        
        # AI Calculation (Calculating stress for every point in the model)
        agent = StressAgent()
        points = torch.tensor(mesh.points).float()
        with torch.no_grad():
            prediction = agent(points).numpy().flatten()
        
        # Add the AI results to the 3D mesh
        mesh["Stress"] = prediction * (load / 1000)
        
        # Show in Website
        plotter = pv.Plotter(window_size=[800, 600])
        plotter.add_mesh(mesh, scalars="Stress", cmap="jet", interpolation='pink')
        stpyvista(plotter)
        st.success("Calculation Complete! Red areas indicate maximum stress.")
