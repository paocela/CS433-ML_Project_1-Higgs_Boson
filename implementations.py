import numpy as np
from helpers.py import compute_loss

"""Least Squares"""

def least_squares(y, tx):
    # Grad for MSE loss function: 1/N*XT(Xw-Y). This we set = 0.
    # => we should solve XTXW = XTY
    # instead of using inv (computationally heavy, inverse of XTX = inverse of DxD), we will solve the equation using np.solve(A,b)
    # where Ax=b form means A=XTX (Rdxd), b=XTY (Rd)
    
    # calculate parameters of linear system
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    # solve linear system and compute loss
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    return w, loss



"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    grad = -1/(y.shape[0])*(tx.T@(y-tx@w))
    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        # update w by the gradient
        w = w - gamma*gradient
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss



"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    
    grad = -1/(y.shape[0])*(tx.T@(y-tx@w))
    return grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size=1, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            
            # update w by gradient
            w = w - gamma*gradient
            
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss



"""Ridge Regression"""

def ridge_regression(y, tx, lambda_):
    # calculate parameters of linear system
    a = (tx.T @ tx) + ((2 * y.shape[0] * lambda_) * np.eye(tx.shape[1]))
    b = tx.T @ y
    
    # solve linear system and compute loss
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w_ridge)

    return w, loss
