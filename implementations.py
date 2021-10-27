import numpy as np
import datetime
from helpers import *
from plots import *

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
        
        print("Progression... ({bi}/{ti})".format(bi=ind, ti=lambdas.shape[0]))
        # ridge regression with a given lambda
        weights_train, mse_train = ridge_regression(y_train, matrix_train, lambda_)
        
        rmse_tr.append(np.sqrt(2 * mse_train))
        rmse_te.append(np.sqrt(2 * compute_loss(y_test, matrix_test, weights_train)))
        
    
    index_lambda_optimal = np.argmin(rmse_te)
    best_lambda = lambdas[index_lambda_optimal]
    
    print(f"Ridge Regression Hypherparameter found: Lambda = {best_lambda}")
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)
    
    return best_lambda
    

def ridge_regression(y, tx, lambda_):
    """Ridge Regression Algorithm"""
    # calculate parameters of linear system
    a = (tx.T @ tx) + ((2 * y.shape[0] * lambda_) * np.eye(tx.shape[1]))
    b = tx.T @ y
    
    # solve linear system and compute loss
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    
    return w, loss

##########################################

"""K-fold cross validation"""

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train:
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    x_test = x[te_indice]
    x_train = x[tr_indice]
    
    # form data with polynomial degree: 
    matrix_train = build_poly(x_train, degree) # matrix = tx
    matrix_test = build_poly(x_test, degree) # matrix = tx
    
    # ridge regression: 
    weights_train, mse_train = ridge_regression(y_train, matrix_train, lambda_)
    mse_test = compute_loss(y_test, matrix_test, weights_train)
   
    return mse_train, mse_test

def find_lambda_cross_validation(y, x, degree, ratio, seed, lambdas, k_fold):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation:
    for ind, lambda_ in enumerate(lambdas):
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            mse_train, mse_test = cross_validation(y, x, k_indices, k, lambda_, degree)
            
            rmse_tr_tmp.append(np.sqrt(2 * mse_train))
            rmse_te_tmp.append(np.sqrt(2 * mse_test))
        
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
"""Cross-validation for least squares""" 
 
def cross_validation(y, x, k_indices, k, lambda_, degree): 
    """return the loss of ridge regression.""" 
    # *********************************************** 
    # get k'th subgroup in test, others in train: 
    # *********************************************** 
    x_train, y_train = x[np.ravel(k_indices[: k]), :], y[np.ravel(k_indices[: k])] 
    x_test, y_test = x[k_indices[k]], y[k_indices[k]] 
    x_train_temp = x[np.ravel(k_indices[k+1 :]), :] 
    y_train_temp = y[np.ravel(k_indices[k+1 :])] 
    x_train, y_train = np.append(x_train, x_train_temp, 0), np.append(y_train, y_train_temp) 
 
    # *********************************************** 
    # form data with polynomial degree: Not done here 
    # *********************************************** 
    #polx_train = build_poly(x_train, degree) 
    #polx_test = build_poly(x_test, degree) 
     
    # *********************************************** 
    # find least squares 
    # *********************************************** 
    weights, mse_train = least_squares(y_train, x_train) 
                              
    # *********************************************** 
    # calculate the loss for train and test data: 
    # *********************************************** 
    mse_test = compute_loss(y_test, x_test, weights) 
                              
    loss_tr = mse_train 
    loss_te = mse_test 
     
    return loss_tr, loss_te 
 
def build_k_indices(y, k_fold, seed): 
    """build k indices for k-fold.""" 
    num_row = y.shape[0] 
    interval = int(num_row / k_fold) 
    np.random.seed(seed) 
    indices = np.random.permutation(num_row) 
    k_indices = [indices[k * interval: (k + 1) * interval] 
                 for k in range(k_fold)] 
    return np.array(k_indices)

def cross_validation_least_squares(y, x, degree, seed, lambda_, k_fold):
    # split data in k fold 
    k_indices = build_k_indices(y, k_fold, seed) 
    # define lists to store the loss of training data and test data 
    rmse_tr = [] 
    rmse_te = [] 
    # *********************************************** 
    # cross validation: 

    loss_tr = [] 
    loss_te = [] 

    for k in range(k_fold): 
        loss_train, loss_test = cross_validation(y, x, k_indices, k, lambda_, degree) 
        loss_tr.append(loss_train) 
        loss_te.append(loss_test) 
    rmse_tr.append(sum(loss_tr)/k_fold) 
    rmse_te.append(sum(loss_te)/k_fold) 

    print(rmse_tr, rmse_te)
    
##########################################

'''
logistic regression
'''

def sigmoid(t):
    """apply the sigmoid function on t."""
    #sigmoid = (1+np.exp(-t))**(-1)
    # sigmoid = np.exp(-np.logaddexp(0, -t))
    print(t)
    return 1.0 / (1 + np.exp(-t))

def calculate_log_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    y = y.reshape((y.shape[0], 1))
    grad = tx.T.dot(pred - y)
    return grad


def log_learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = calculate_log_loss(y, tx, w)

    grad = calculate_log_gradient(y, tx, w)

    w = w - gamma * grad
    
    
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_log_loss(y, tx, w) + lambda_/2*np.linalg.norm(w)
    gradient = calculate_log_gradient(y, tx, w) + lambda_/2*2*w #gradient of euclnorm(x) = 2x
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*gradient
    
    
    return loss, w

