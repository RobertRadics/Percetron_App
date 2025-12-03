import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_data(self, X, y, title="Adatpontok megjelenítése"):
        plt.figure(figsize=(10, 8))
        
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        plt.scatter(class_0[:, 0], class_0[:, 1], c='red', marker='o', 
                   label='Osztály 0', s=100, alpha=0.7, edgecolors='black')
        plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', marker='s', 
                   label='Osztály 1', s=100, alpha=0.7, edgecolors='black')
        
        plt.xlabel('X1', fontsize=12)
        plt.ylabel('X2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()