import numpy as np
import datetime
from helpers import *

"""Loss"""

def compute_loss(y, tx, w):
    prediction = np.dot(tx, w)
    errors = prediction - y
    return np.mean(errors ** 2)


##########################################

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

##########################################

"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    # error = y - prediction
    # prediction = Xtilda * w
    grad = -1/(y.shape[0])*(tx.T@(y-tx@w))
    
    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss and gradient
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        # update weight by gradient
        w = w - (gamma * gradient)
        
        # store weight and loss
        ws.append(w)
        losses.append(loss)
        
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = gradient_descent(y, tx, initial_w, max_iters, gamma)
    end_time = datetime.datetime.now()
    
    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))
    
    print(f"Gradient Descend: final loss = {gradient_losses[-1]}")
    
    # return only final loss and optimal weights
    return gradient_ws[-1], gradient_losses[-1]

##########################################

"""Stochastic Gradient Descent"""

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    grad = -1/(y.shape[0])*(tx.T@(y-tx@w))
    return grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            
            stoc_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            
            w = w - (gamma * stoc_gradient)
            loss = compute_loss(y, tx, w)
            
            ws.append(w)
            losses.append(loss)
            #print("Stocastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1
    
    # Start SGD
    start_time = datetime.datetime.now()
    sgd_losses, sgd_ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
    
    print(f"Gradient Descend: final loss = {sgd_losses[-1]}")
    
    # return only final loss and optimal weights
    return sgd_ws[-1], sgd_losses[-1]

##########################################

"""Ridge Regression"""

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def find_ridge_hyperparameters(y, x, degree, ratio, seed, lambdas):
    """function to find best lambda parameter for ridge regression"""
    
    # split the data, and return train and test data
    x_train, x_test, y_train, y_test = split_data(x, y, ratio, seed)

    # form train and test data with polynomial basis function
    matrix_train = build_poly(x_train, degree) # matrix = tx
    matrix_test = build_poly(x_test, degree) # matrix = tx
    
    rmse_tr = []
    rmse_te = []

    for ind, lambda_ in enumerate(lambdas):
        # ridge regression with a given lambda
        weights_train, mse_train = ridge_regression(y_train, matrix_train, lambda_)
        
        rmse_tr.append(np.sqrt(2 * mse_train))
        rmse_te.append(np.sqrt(2 * compute_loss(y_test, matrix_test, weights_train)))
        
    
    index_lambda_optimal = np.argmin(rmse_te)
    best_lambda = lambdas[index_lambda_optimal]
    
    print(f"Ridge Regression Hypherparameter found: Lambda = {best_lambda}")

    return best_lambda
    

def ridge_regression(y, tx, lambda_):
    """Ridge Regression Algorithm"""
    # calculate parameters of linear system
    a = (tx.T @ tx) + ((2 * y.shape[0] * lambda_) * np.eye(tx.shape[1]))
    b = tx.T @ y
    
    # solve linear system and compute loss
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    print(f"Ridge Regression: loss = {loss}")

    return w, loss
