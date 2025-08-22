### Import Packages ###
import numpy as np
import pandas as pd

def GenerateBurbridgeData(n_samples=500, delta=0.0, sigma_epsilon=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    ### Generate inputs from N(\mu=0.2, \sd^2=0.4^2) ###
    x = np.random.normal(loc=0.2, scale=0.4, size=n_samples)
    
    ### Calculate z and r(x) ###
    z = (x - 0.2) / 0.4
    r_x = (z**3 - 3 * z) / np.sqrt(6)
    
    ### Calculate the deterministic part of the function ###
    f_x = 1 - x + x**2 + delta * r_x
    
    ### Add noise ###
    epsilon = np.random.normal(loc=0, scale=sigma_epsilon, size=n_samples)
    y = f_x + epsilon
    
    ### Create and return the final DataFrame ###
    df = pd.DataFrame({'X1': x, 'Y': y})
    return df