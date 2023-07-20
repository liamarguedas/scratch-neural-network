import numpy as np

def train(X, y, weights, biases, observations, learning_rate = 0.02, epochs = 100, verbose = True, return_outputs = True):
    
    for epoch in range(epochs):
        
        outputs = np.dot(X, weights) + biases
        
        deltas = outputs - y
        
        loss = (np.sum(deltas ** 2) / 2) / observations
        
        if verbose:
            
            print(f"Epoch {epoch + 1}/{epochs}: loss {loss}")
            
        deltas_scaled = deltas / observations
        
        weights = weights - learning_rate * np.dot(X.T, deltas_scaled)
        
        biases = biases - learning_rate * np.sum(deltas_scaled)
    
    if return_weights_biases:    
    
        return (weights, biases), outputs