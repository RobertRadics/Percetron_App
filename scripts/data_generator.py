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

def generate_and_save_example_data(filename="example_data.csv", n_samples=100):
    df = generate_linear_separable_data(n_samples=n_samples, random_state=42)
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / filename
    df.to_csv(file_path, index=False)
    print(f"Példa adatfájl létrehozva: {file_path}")
    print(f"Adatpontok száma: {len(df)}")
    return df


if __name__ == "__main__":
    generate_and_save_example_data("example_data.csv", n_samples=100)