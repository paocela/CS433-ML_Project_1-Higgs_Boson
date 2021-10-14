import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    prediction = np.dot(tx, w)
    errors = prediction - y
    return np.mean(errors ** 2)
