#Normal
def least_squares(y, tx):
    """calculate the least squares solution."""
    # Grad for MSE loss function: 1/N*XT(Xw-Y). This we set = 0.
    # => we should solve XTXW = XTY
    # instead of using inv (computationally heavy, inverse of XTX = inverse of DxD), we will solve the equation using np.solve(A,b)
    # where Ax=b form means A=XTX (Rdxd), b=XTY (Rd)
    
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    mse = 1/y.shape[0]*np.linalg.norm(tx@w-y)**2
    return mse, w
    


#GD
def compute_gradient(y, tx, w):
    grad = -1/(y.shape[0])*(tx.T@(y-tx@w))
    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

#SGD
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    grad = -1/(y.shape[0])*(tx.T@(y-tx@w))
    return grad



def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        #så börja med att slumpa batchen med hjälp av helpers.py:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            #gradient: skicka in en/flera random x:s och y:s och w:s från batch slump
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            #samma för lossen?
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            
            w = w - gamma*gradient

            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # ridge regression:
    # ***************************************************
    a = (tx.T @ tx) + ((2 * y.shape[0] * lambda_) * np.eye(tx.shape[1]))
    b = tx.T @ y

    
    w_ridge = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w_ridge)

    return w_ridge, mse
