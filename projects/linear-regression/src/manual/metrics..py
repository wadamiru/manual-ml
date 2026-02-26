def mse(y, y_hat):
    """
    Mean Squared Error
    """
    return ((y - y_hat) ** 2).mean()