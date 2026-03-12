import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()

    def forward(self, x):
        """
        Compute y_hat = w*x + b
        """
        return self.w * x + self.b