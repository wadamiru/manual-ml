import numpy as np

def fit_ols(X, y):
    """
    Ordinary least squares via stable least-squares solver (SVD).
    """
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)

    return theta

def predict(X, theta):
    """
    Linear model prediction.
    """
    return X @ theta