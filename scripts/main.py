import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perceptron import Perceptron
from src.data_loader import DataLoader
from src.visualizer import Visualizer
import numpy as np

def print_menu():
    print("\n" + "="*60)
    print("INTERAKTÍV PERCEPTRON TANÍTÓ ALKALMAZÁS")
    print("="*60)
    print("1. Adatfájl betöltése")
    print("2. Adatok megjelenítése")
    print("3. Perceptron inicializálása")
    print("4. Egy lépés tanítás")
    print("5. Teljes tanítás futtatása")
    print("5a. Interaktív tanítás (epochonkénti megállás)")
    print("6. Döntési határ megjelenítése")
    print("7. Konvergencia görbe megjelenítése")
    print("8. Paraméterek beállítása")
    print("9. Kilépés")
    print("="*60)
    
def load_data_interactive():
    loader = DataLoader()
    
    print("\n--- Adatfájl betöltése ---")
    print("Elérhető adatfájlok:")
    print("  - data.csv (alapértelmezett)")
    print("  - data_easy.csv (könnyen szeparálható)")
    print("  - data_hard.csv (nehezebben szeparálható)")
    print("  - data_vertical.csv (függőleges elválasztás)")
    print("  - data_diagonal.csv (átlós elválasztás)")
    print("\nAdja meg a CSV fájl nevét (vagy nyomjon Enter-t az alapértelmezett 'data.csv' használatához):")
    filepath = input("Fájlnév: ").strip()
    
    if not filepath:
        filepath = "data.csv"
    
    if not filepath.endswith('.csv'):
        filepath += '.csv'
    
    if not os.path.isabs(filepath):
        data_dir = Path(__file__).parent.parent / 'data'
        data_file = data_dir / filepath
        if data_file.exists():
            filepath = str(data_file)
        else:
            current_dir_file = os.path.join(os.getcwd(), filepath)
            if os.path.exists(current_dir_file):
                filepath = current_dir_file
            else:
                filepath = str(data_file)
    
    if loader.load_from_file(filepath):
        loader.print_summary()
        return loader.get_data()
    else:
        return None, None
    
def initialize_perceptron(X, learning_rate=0.01):
    n_features = X.shape[1]
    perceptron = Perceptron(learning_rate=learning_rate, n_features=n_features)
    print(f"\nPerceptron inicializálva:")
    print(f"  Tanulási ráta: {learning_rate}")
    print(f"  Kezdeti súlyok: {perceptron.weights}")
    print(f"  Kezdeti bias: {perceptron.bias:.3f}")
    return perceptron

def train_step_interactive(perceptron, X, y):
    print("\n--- Egy lépés tanítás ---")
    errors = perceptron.fit_step(X, y)
    print(f"Hibák száma ebben a lépésben: {errors}")
    print(f"Aktuális súlyok: {perceptron.weights}")
    print(f"Aktuális bias: {perceptron.bias:.3f}")
    return errors

def train_full_interactive(perceptron, X, y):
    print("\n--- Teljes tanítás ---")
    print("Adja meg a maximális epoch számot (Enter = 100):")
    max_epochs_input = input("Max epoch: ").strip()
    max_epochs = int(max_epochs_input) if max_epochs_input else 100
    
    print(f"\nTanítás indítása {max_epochs} epoch-ra...")
    errors_history = perceptron.fit(X, y, max_epochs=max_epochs, verbose=True)
    
    print(f"\nTanítás befejezve!")
    print(f"Összesen {len(errors_history)} epoch futott le.")
    if errors_history[-1] == 0:
        print("Konvergencia elérve - nincs hiba!")
    else:
        print(f"Még mindig vannak hibák: {errors_history[-1]}")
    
    return errors_history