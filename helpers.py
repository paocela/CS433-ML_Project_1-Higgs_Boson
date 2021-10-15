import numpy as np

"""Function used to compute the loss."""
def compute_loss(y, tx, w):
    prediction = np.dot(tx, w)
    errors = prediction - y
    return np.mean(errors ** 2)


