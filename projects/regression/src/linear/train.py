import numpy as np
from metrics import mse

def train(model, x, y, lr=0.01, epochs=1000, verbose=True):
    for e in range(epochs):
        # forward
        y_hat = model.forward(x)
        loss = mse(y, y_hat)

        # gradients (d/dw, d/db of MSE)
        dw = -2 * np.mean((y - y_hat) * x)
        db = -2 * np.mean(y - y_hat)

        # update (SGD)
        model.w -= lr * dw
        model.b -= lr * db

        if verbose and e % 100 == 0:
            print(f"Epoch {e:4d} | Loss: {loss:.6f}")

    return model