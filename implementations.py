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

"""Logistic Regression"""

def sigmoid(t): 
    """apply the sigmoid function on t.""" 
    #sigmoid = (1+np.exp(-t))**(-1) 
    # sigmoid = np.exp(-np.logaddexp(0, -t)) 
    #print(t) 
    return 1.0 / (1 + np.exp(-t)) 
 
def calculate_log_loss(y, tx, w): 
    """compute the loss: negative log likelihood.""" 
    pred = sigmoid(tx.dot(w)) 
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)) 
    sum_loss = np.squeeze(- loss) 
    return sum_loss/y.shape[0] 
 
def calculate_log_gradient(y, tx, w): 
    """compute the gradient of loss.""" 
    #print('tx.dot(w)', tx.dot(w)) 
    pred = sigmoid(tx.dot(w)) 
    #print('sigm(tx.dot(w)', pred, 'y', y) 
    #print('tx.T', tx.T) 
    grad = tx.T@(pred - y) 
    return grad/y.shape[0] 
 
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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-5
    losses = [] 
    
    # y_0 = y but -1 -> 0 instead 
    y_tmp = np.ones(len(y_pp)) 
    y_tmp[np.where(y_pp==-1)] = 0 
    print(y, y_tmp) 


    np.set_printoptions(suppress=True, floatmode = 'fixed') 
    
    
    w = initial_w
    # start the logistic regression 
    for iter in range(max_iter): 
        # get loss and update w. 
        loss, w = log_learning_by_gradient_descent(y_tmp, tx_pp, w, gamma) 
        # log info 
        if iter % 10 == 0: 
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss)) 
        # converge criterion 
        losses.append(loss) 
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold: 
            breaktx = np.c_[np.ones((y.shape[0], 1)), x]
            
    return loss, w
            
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8 
    losses = [] 


    # y_0 = y but -1 -> 0 instead 
    y_0 = np.ones(len(y_pp)) 
    y_0[np.where(y_pp==-1)] = 0 
    print(y, y_0) 

    w = initial_w
    # start the logistic regression 
    for iter in range(max_iter): 
        # get loss dand update w. 
        loss, w = learning_by_penalized_gradient(y_0, tx, w, gamma, lambda_) 
        # log info 
        if iter % 1 == 0: 
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss)) 
        # converge criterion 
        losses.append(loss) 
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold: 
            break
    
    return loss, w

##########################################

"""K-fold cross validation for all methods used"""
"""
   - least squares
   - gradient descent
   - stochastic gradient descent
   - ridge regression
   - logistic regression
   - regularized logistic regression
"""
  
def cross_validation(model_used, y, x, k_indices, k, seed = 0, max_iters = 0, initial_w = 0, lambda_ = 0, gamma = 0, degree = 0):  
    # cross validation for model_used = 'least_squares' or 'gradient_descent' or 'stochastic_gradient_descent' or 'ridge_regression' 
     
    # get k'th subgroup in test, others in train:  
    x_train, y_train = x[np.ravel(k_indices[: k]), :], y[np.ravel(k_indices[: k])]  
    x_test, y_test = x[k_indices[k]], y[k_indices[k]]  
    x_train_temp = x[np.ravel(k_indices[k+1 :]), :]  
    y_train_temp = y[np.ravel(k_indices[k+1 :])]  
    x_train, y_train = np.append(x_train, x_train_temp, 0), np.append(y_train, y_train_temp)  
  
    # form data with polynomial degree: only for ridge regression  
    if degree > 0: 
        x_train = build_poly(x_train, degree)  
        x_test = build_poly(x_test, degree)  
      
    f model_used == 'least_squares': 
        weights, loss_train = least_squares(y_train, x_train) 
    elif model_used == 'gradient_descent': 
        weights, loss_train = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma) 
    elif model_used == 'stochastic_gradient_descent': 
        weights, loss_train = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma) 
    elif model_used == 'ridge_regression': 
        weights, loss_train = ridge_regression(y_train, x_train, lambda_) 
    elif model_used == 'logistic_regression' 
         
        weights, loss_train = logistic_regression(y_train, x_train, initial_w, max_iters, gamma) 
    #elif model_used == 'reg_logistic_regression' 
    #    weights, mse_train =  
                            
     
    # *******************************************  
    # calculate the loss for train and test data:  
    # ******************************************* 
    if model_used == 'logistic_regression' or 'reg_logistic_regression': 
        #loss_test =  
    else: loss_test = compute_loss(y_test, x_test, weights)  
                               
    #loss_tr = mse_train  
    #loss_te = mse_test  
      
    return loss_train, loss_test  
  
def build_k_indices(y, k_fold, seed):  
    """build k indices for k-fold."""  
    num_row = y.shape[0]  
    interval = int(num_row / k_fold)  
    np.random.seed(seed)  
    indices = np.random.permutation(num_row)  
    k_indices = [indices[k * interval: (k + 1) * interval]  
                 for k in range(k_fold)]  
    return np.array(k_indices) 
 
def evaluate_using_cross_validation(model_used, y, x, k_fold, seed = 0, max_iters = 0, initial_w = 0, lambda_ = 0, gamma = 0, degree = 0): 
    # model_used = 'least_squares' or 'gradient_descent' or 'stochastic_gradient_descent' or 'ridge_regression' 
     
    # split data in k fold  
    k_indices = build_k_indices(y, k_fold, seed)  
    # define lists to store the loss of training data and test data  
    mse_tr = []  
    mse_te = []  
    # *******************************************  
    # cross validation:  
 
    loss_tr = []
    loss_te = []  
 
    for k in range(k_fold):  
        loss_train, loss_test = cross_validation(model_used, y, x, k_indices, k, seed, max_iters, initial_w, lambda_, gamma, degree)  
        loss_tr.append(loss_train)  
        loss_te.append(loss_test)  
    mse_tr.append(sum(loss_tr)/k_fold)  
    mse_te.append(sum(loss_te)/k_fold)  
 
    print('-->  ',model_used, ' cross-validation: avg_mse_tr=', mse_tr, ' avg_mse_te=', mse_te) 
     
    
