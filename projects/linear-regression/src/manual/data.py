import numpy as np

def make_data(w, b, nsamples=100, noise=False, noise_std=1.0):
    """
    Generate linear data: y = w*x + b (+ optional Gaussian noise)
    """
    x = np.random.rand(nsamples)
    y = w * x + b
    
    if noise:
        eps = np.random.normal(0, noise_std, size=nsamples)
        y += eps

    return x, y