import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perceptron import Perceptron
from src.data_loader import DataLoader

st.set_page_config(
    page_title="Perceptron Tanító",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'perceptron' not in st.session_state:
    st.session_state.perceptron = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0

def reset_training():
    st.session_state.training_history = []
    st.session_state.current_epoch = 0
    if st.session_state.perceptron:
        st.session_state.perceptron.errors_history = []
        st.session_state.perceptron.weights_history = []

def plot_data_interactive(X, y):
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['Osztály'] = y.astype(str)
    df['Osztály'] = df['Osztály'].replace({'0': 'Osztály 0', '1': 'Osztály 1'})
    
    fig = px.scatter(
        df, x='X1', y='X2', color='Osztály',
        color_discrete_map={'Osztály 0': '#ef4444', 'Osztály 1': '#3b82f6'},
        title='Adatpontok megjelenítése',
        labels={'X1': 'X1', 'X2': 'X2'},
        hover_data=['X1', 'X2']
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    fig.update_layout(
        width=800,
        height=600,
        title_font_size=20,
        font_size=12
    )
    return fig

def plot_decision_boundary_interactive(X, y, perceptron, epoch=None):
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=class_0[:, 0], y=class_0[:, 1],
        mode='markers',
        name='Osztály 0',
        marker=dict(size=12, color='#ef4444', line=dict(width=1, color='black'))
    ))
    
    fig.add_trace(go.Scatter(
        x=class_1[:, 0], y=class_1[:, 1],
        mode='markers',
        name='Osztály 1',
        marker=dict(size=12, color='#3b82f6', line=dict(width=1, color='black'))
    ))
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    x1_boundary, x2_boundary = perceptron.get_decision_boundary((x1_min, x1_max))
    
    mask = (x2_boundary >= x2_min) & (x2_boundary <= x2_max)
    
    fig.add_trace(go.Scatter(
        x=x1_boundary[mask], y=x2_boundary[mask],
        mode='lines',
        name='Döntési határ',
        line=dict(color='#10b981', width=3)
    ))
    
    title = 'Perceptron döntési határa'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    
    fig.update_layout(
        title=title,
        xaxis_title='X1',
        yaxis_title='X2',
        width=800,
        height=600,
        title_font_size=20,
        font_size=12,
        xaxis=dict(range=[x1_min, x1_max]),
        yaxis=dict(range=[x2_min, x2_max])
    )
    
    return fig

def plot_convergence_interactive(errors_history):
    epochs = list(range(1, len(errors_history) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=errors_history,
        mode='lines+markers',
        name='Hibák száma',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, color='#3b82f6')
    ))
    
    fig.update_layout(
        title='Konvergencia görbe - Hibák száma epochonként',
        xaxis_title='Epoch',
        yaxis_title='Hibák száma',
        width=800,
        height=400,
        title_font_size=20,
        font_size=12,
        hovermode='x unified'
    )
    
    return fig

data_dir = Path(__file__).parent.parent / 'data'
data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')] if data_dir.exists() else []

st.markdown('<h1 class="main-header">Perceptron Tanító Alkalmazás</h1>', unsafe_allow_html=True)