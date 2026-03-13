import numpy as np

def make_polynomial_data(nsamples=100, deg=3):
    """
    Generate synthetic polynomial regression data.

    Returns:
        xpoly : (nsamples, deg+1) polynomial feature matrix [1, x, x^2, ...]
        y     : noisy polynomial target
    """

    # --- sample inputs and initialise noisy target --- #
    x = np.linspace(-3, 3, nsamples)
    eps = np.random.randn(nsamples) *2
    y = eps

    # --- construct random polynomial signal --- #
    for d in range(deg):
        y += np.random.randn() *x **d

    # --- build polynomial feature matrix --- #
    xpoly = np.ones((nsamples, deg +1))
    for d in range(deg +1):
        xpoly[:, d] = x **d

    return xpoly, y