import numpy as np

def generate_weights(init_range = 0.1, return_biases = False):
    
    weights = np.random.uniform(low = -init_range, high = init_range, size = (2, 1))

    if return_biases:
        
        return weights, np.random.uniform(low = -init_range, high = init_range, size = 1)
    
    return weights