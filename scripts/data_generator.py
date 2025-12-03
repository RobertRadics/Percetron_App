import numpy as np
import pandas as pd
from pathlib import Path

def generate_linear_separable_data(n_samples=100, n_features=2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_features != 2:
        raise ValueError("A generátor jelenleg csak 2D adatokat tud generálni.")
    
    n_class0 = n_samples // 2
    X_class0 = np.random.randn(n_class0, n_features) * 0.5
    X_class0[:, 1] -= 1.0
    
    n_class1 = n_samples - n_class0
    X_class1 = np.random.randn(n_class1, n_features) * 0.5
    X_class1[:, 1] += 1.0
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
    
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = y.astype(int)
    
    return df