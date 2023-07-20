import numpy as np

def generate_data(observations = 1_000, return_targets = False):
    
    xs = np.random.uniform(low = -10, high = 10, size = (observations, 1))
    
    zs = np.random.uniform(low = -10, high = 10, size = (observations, 1))
    
    noise = np.random.uniform(low = -1, high = 1, size = (observations, 1))
    
    if return_targets:
    
        return np.column_stack((xs, zs)), (2 * xs) - (3 * zs) + 5 + noise
    
    return np.column_stack((xs, zs))