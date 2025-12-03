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

with st.sidebar:
    st.header("Beállítások")
    
    st.subheader("Adatbetöltés")
    
    selected_file = st.selectbox(
        "Válasszon adatfájlt:",
        options=data_files if data_files else ['data.csv'],
        index=0 if data_files else None
    )
    
if st.button("Adatok betöltése", type="primary", use_container_width=True):
        loader = DataLoader()
        file_path = data_dir / selected_file if selected_file in data_files else data_dir / 'data.csv'
        if loader.load_from_file(str(file_path)):
            st.session_state.X, st.session_state.y = loader.get_data()
            st.session_state.data_loaded = True
            st.session_state.data_info = {
                'samples': len(st.session_state.X),
                'features': st.session_state.X.shape[1],
                'classes': np.unique(st.session_state.y)
            }
            reset_training()
            st.success(f"Adatok sikeresen betöltve: {selected_file}")
            st.rerun()
        else:
            st.error("Hiba történt az adatbetöltés során!")
    
        st.divider()
    
st.subheader("Perceptron Paraméterek")
    
learning_rate = st.slider(
        "Tanulási ráta (Learning Rate)",
        min_value=0.001,
        max_value=0.5,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
    
max_epochs = st.number_input(
        "Maximális epoch szám",
        min_value=1,
        max_value=1000,
        value=100,
        step=10
    )

if st.button("Perceptron inicializálása", use_container_width=True):
        if st.session_state.data_loaded:
            st.session_state.perceptron = Perceptron(
                learning_rate=learning_rate,
                n_features=st.session_state.X.shape[1]
            )
            reset_training()
            st.success("Perceptron inicializálva!")
            st.rerun()
        else:
            st.warning("Először töltse be az adatokat!")
    
st.divider()
    
if st.button("Minden törlése", use_container_width=True):
        st.session_state.perceptron = None
        st.session_state.X = None
        st.session_state.y = None
        st.session_state.data_loaded = False
        reset_training()
        st.rerun()

if st.session_state.data_loaded:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Adatpontok száma", st.session_state.data_info['samples'])
    with col2:
        st.metric("Attribútumok száma", st.session_state.data_info['features'])
    with col3:
        st.metric("Osztályok", len(st.session_state.data_info['classes']))
    with col4:
        if st.session_state.perceptron:
            st.metric("Perceptron", "Inicializálva")
        else:
            st.metric("Perceptron", "Nincs inicializálva")

tab1, tab2, tab3, tab4 = st.tabs(["Adatok", "Tanítás", "Vizualizáció", "Konvergencia"])
    
with tab1:
        st.subheader("Adatpontok megjelenítése")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            fig_data = plot_data_interactive(st.session_state.X, st.session_state.y)
            st.plotly_chart(fig_data, use_container_width=True, key="data_plot")
        
        with col_right:
            st.subheader("Adatstatisztikák")
            df_data = pd.DataFrame(st.session_state.X, columns=[f'X{i+1}' for i in range(st.session_state.X.shape[1])])
            df_data['Label'] = st.session_state.y
            st.dataframe(df_data.describe(), use_container_width=True)
            
            st.subheader("Osztály eloszlás")
            class_counts = pd.Series(st.session_state.y).value_counts().sort_index()
            st.bar_chart(class_counts)
            
with tab2:
        st.subheader("Perceptron Tanítás")
        
        if not st.session_state.perceptron:
            st.warning("Először inicializálja a perceptront az oldalsávban!")
        else:
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Tanulási ráta", f"{st.session_state.perceptron.learning_rate:.3f}")
            with col_info2:
                if st.session_state.perceptron.errors_history:
                    st.metric("Utolsó epoch hibái", st.session_state.perceptron.errors_history[-1])
                else:
                    st.metric("Utolsó epoch hibái", "N/A")
            with col_info3:
                st.metric("Epoch-ok száma", len(st.session_state.perceptron.errors_history))
            
            st.subheader("Aktuális paraméterek")
            col_w1, col_w2, col_bias = st.columns(3)
            
            with col_w1:
                st.metric("Súly 1 (w1)", f"{st.session_state.perceptron.weights[0]:.4f}")
            with col_w2:
                st.metric("Súly 2 (w2)", f"{st.session_state.perceptron.weights[1]:.4f}")
            with col_bias:
                st.metric("Bias (b)", f"{st.session_state.perceptron.bias:.4f}")
            
            st.divider()
            
            col_train1, col_train2, col_train3 = st.columns(3)
            
            with col_train1:
                if st.button("Egy lépés tanítás", use_container_width=True, type="primary"):
                    if st.session_state.perceptron and st.session_state.X is not None:
                        errors = st.session_state.perceptron.fit_step(st.session_state.X, st.session_state.y)
                        st.session_state.perceptron.errors_history.append(errors)
                        st.session_state.perceptron.weights_history.append(
                            (st.session_state.perceptron.weights.copy(), st.session_state.perceptron.bias)
                        )
                        st.session_state.current_epoch += 1
                        
                        if errors == 0:
                            st.success(f"Konvergencia elérve! {st.session_state.current_epoch} epoch után nincs hiba!")
                        else:
                            st.info(f"Epoch {st.session_state.current_epoch}: {errors} hiba")
                        st.rerun()
                        with col_train2:
                            if st.button("Teljes tanítás", use_container_width=True, type="primary"):
                                if st.session_state.perceptron and st.session_state.X is not None:
                                    progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        st.session_state.perceptron.errors_history = []
                        st.session_state.perceptron.weights_history = []
                        st.session_state.current_epoch = 0
                        
                        for epoch in range(max_epochs):
                            errors = st.session_state.perceptron.fit_step(st.session_state.X, st.session_state.y)
                            st.session_state.perceptron.errors_history.append(errors)
                            st.session_state.perceptron.weights_history.append(
                                (st.session_state.perceptron.weights.copy(), st.session_state.perceptron.bias)
                            )
                            st.session_state.current_epoch += 1
                            
                            progress = (epoch + 1) / max_epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch + 1}/{max_epochs}: {errors} hiba")
                            
                            if errors == 0:
                                st.success(f"Konvergencia elérve! {epoch + 1} epoch után nincs hiba!")
                                break
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.rerun()
            