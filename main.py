# main.py – App Entry Point
import streamlit as st
from ui.ui import render_ui

st.set_page_config(page_title="Fusion Mesh Simulator", layout="wide")
render_ui()
