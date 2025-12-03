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