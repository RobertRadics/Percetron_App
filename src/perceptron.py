import numpy as np

class Perceptron:
    
    def __init__(self, learning_rate=0.01, n_features=2):
        self.learning_rate = learning_rate
        self.n_features = n_features
        
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        
        self.bias = np.random.uniform(-0.5, 0.5)
        
        self.errors_history = []
        self.weights_history = []
        
    def activation_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        if X.ndim == 1:
            linear_output = np.dot(X, self.weights) + self.bias
            return self.activation_function(linear_output)
        
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation_function(x) for x in linear_output])