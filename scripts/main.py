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

def train_interactive_step_by_step(perceptron, X, y, visualizer):
    print("\n--- Interaktív tanítás (epochonkénti megállás) ---")
    print("Adja meg a maximális epoch számot (Enter = 100):")
    max_epochs_input = input("Max epoch: ").strip()
    max_epochs = int(max_epochs_input) if max_epochs_input else 100
    
    print("\nMinden epoch után dönthet:")
    print("  - Enter: következő epoch")
    print("  - 'g': grafikon megjelenítése")
    print("  - 'q': kilépés")
    
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        
        errors = perceptron.fit_step(X, y)
        
        perceptron.errors_history.append(errors)
        perceptron.weights_history.append((perceptron.weights.copy(), perceptron.bias))
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{max_epochs}")
        print(f"{'='*60}")
        print(f"Hibák száma: {errors}")
        print(f"Súlyok: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
        print(f"Bias: {perceptron.bias:.4f}")
        
        if epoch > 1:
            prev_weights, prev_bias = perceptron.weights_history[-2]
            weight_change = perceptron.weights - prev_weights
            bias_change = perceptron.bias - prev_bias
            print(f"\nSúlyok változása: [{weight_change[0]:.4f}, {weight_change[1]:.4f}]")
            print(f"Bias változása: {bias_change:.4f}")
        
        if errors == 0:
            print(f"\n{'='*60}")
            print("KONVERGENCIA ELÉRVE!")
            print(f"A perceptron {epoch} epoch után konvergált - nincs hiba!")
            print(f"{'='*60}")
            break
        
        print(f"\nMit szeretne tenni? (Enter = folytatás, 'g' = grafikon, 'q' = kilépés)")
        user_input = input("Választás: ").strip().lower()
        
        if user_input == 'q':
            print(f"\nTanítás megszakítva {epoch} epoch után.")
            break
        elif user_input == 'g':
            print("\nGrafikon megjelenítése...")
            fig = visualizer.plot_with_decision_boundary(
                X, y, perceptron, epoch=epoch, show_history=True
            )
            visualizer.show()
            print("\nFolytatás? (Enter = igen, 'q' = kilépés)")
            continue_input = input("Választás: ").strip().lower()
            if continue_input == 'q':
                print(f"\nTanítás megszakítva {epoch} epoch után.")
                break
    
    print(f"\n{'='*60}")
    print("Interaktív tanítás befejezve!")
    print(f"Összesen {epoch} epoch futott le.")
    if perceptron.errors_history and perceptron.errors_history[-1] == 0:
        print("Konvergencia elérve!")
    else:
        print(f"Utolsó epoch hibái: {perceptron.errors_history[-1] if perceptron.errors_history else 'N/A'}")
    print(f"{'='*60}")
    
    return perceptron.errors_history