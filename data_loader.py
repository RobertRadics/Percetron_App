import pandas as pd
import numpy as np
import os

class DataLoader:
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_from_file(self, filepath):
        try:
            if not os.path.exists(filepath):
                print(f"Hiba: A fájl nem található: {filepath}")
                return False
            
            self.data = pd.read_csv(filepath)
            
            if len(self.data.columns) < 3:
                print("Hiba: A CSV fájlnak legalább 3 oszlopnak kell lennie (2 attribútum + 1 címke)")
                return False
            
            self.feature_names = self.data.columns[:-1].tolist()
            label_name = self.data.columns[-1]
            
            self.X = self.data[self.feature_names].values
            self.y = self.data[label_name].values
            
            unique_labels = np.unique(self.y)
            if not (set(unique_labels) <= {0, 1}):
                print("Hiba: A címke oszlop csak 0 és 1 értékeket tartalmazhat")
                return False
            
            if self.X.shape[1] < 2:
                print("Hiba: Legalább 2 bemeneti attribútum szükséges")
                return False
            
            print(f"Sikeres betöltés: {len(self.data)} sor, {self.X.shape[1]} attribútum")
            print(f"Attribútumok: {', '.join(self.feature_names)}")
            print(f"Címke oszlop: {label_name}")
            print(f"Osztályok: {unique_labels}")
            
            return True
            
        except pd.errors.EmptyDataError:
            print("Hiba: A CSV fájl üres")
            return False
        except pd.errors.ParserError as e:
            print(f"Hiba: A CSV fájl formátuma hibás: {e}")
            return False
        except Exception as e:
            print(f"Hiba történt a fájl betöltése során: {e}")
            return False