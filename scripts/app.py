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