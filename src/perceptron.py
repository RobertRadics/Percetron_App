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
      
    def fit_step(self, X, y):
        errors = 0
        
        for i in range(len(X)):
            prediction = self.predict(X[i])
            
            if prediction != y[i]:
                errors += 1
                
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
        
        return errors
    
    def fit(self, X, y, max_epochs=100, verbose=True):
        self.errors_history = []
        self.weights_history = []
        
        for epoch in range(max_epochs):
            errors = self.fit_step(X, y)
            
            self.errors_history.append(errors)
            self.weights_history.append((self.weights.copy(), self.bias))
            
            if verbose:
                print(f"Epoch {epoch + 1}/{max_epochs}: Hibák száma = {errors}, "
                      f"Súlyok = {self.weights}, Bias = {self.bias:.3f}")
            
            if errors == 0:
                if verbose:
                    print(f"Konvergencia elérve {epoch + 1} epoch után!")
                break
        
        return self.errors_history