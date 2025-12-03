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
    
    def plot_with_decision_boundary(self, X, y, perceptron, epoch=None, 
                                    show_history=False):
        if show_history and perceptron.errors_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            ax2 = None
        
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        ax1.scatter(class_0[:, 0], class_0[:, 1], c='red', marker='o', 
                   label='Osztály 0', s=100, alpha=0.7, edgecolors='black')
        ax1.scatter(class_1[:, 0], class_1[:, 1], c='blue', marker='s', 
                   label='Osztály 1', s=100, alpha=0.7, edgecolors='black')
        
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        x1_boundary, x2_boundary = perceptron.get_decision_boundary((x1_min, x1_max))
        
        mask = (x2_boundary >= x2_min) & (x2_boundary <= x2_max)
        ax1.plot(x1_boundary[mask], x2_boundary[mask], 'g-', linewidth=2, 
                label='Döntési határ')
        
        ax1.set_xlim(x1_min, x1_max)
        ax1.set_ylim(x2_min, x2_max)
        ax1.set_xlabel('X1', fontsize=12)
        ax1.set_ylabel('X2', fontsize=12)
        
        title = 'Perceptron döntési határa'
        if epoch is not None:
            title += f' - Epoch {epoch}'
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        if ax2 is not None and perceptron.errors_history:
            epochs = range(1, len(perceptron.errors_history) + 1)
            ax2.plot(epochs, perceptron.errors_history, 'b-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Hibák száma', fontsize=12)
            ax2.set_title('Konvergencia görbe', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(epochs[::max(1, len(epochs)//10)])
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, errors_history):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(errors_history) + 1)
        plt.plot(epochs, errors_history, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Hibák száma', fontsize=12)
        plt.title('Konvergencia görbe - Hibák száma epochonként', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()