import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(prediction, data):
    """Form (y,tX) to get regression data in matrix form."""
    y = prediction
    x = data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

"""Substitute NaN values (-999) with mean of each row"""
# TODO: substitute with code more efficient 
# - np.nanmean()
# - arr[arr > 255] = x
def substitute_nan_with_mean(tx):
    tx[tx==-999] = np.nan
    avg_per_column = np.nanmean(tx, axis=0)
    index_to_subst = np.where(np.isnan(tx))
    tx[index_to_subst] = np.take(avg_per_column, index_to_subst[1])
    return tx

"""Substitute outliers using quantile ranges with the median """
def remove_outliers(y, tx, low_bound, high_bound):
    # index columns with outliers visible from histograms
    index_outliers_features = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 17, 20, 22, 24, 27, 28, 29, 30]
    
    # consider only related columns
    tx_outliers = tx[:, index_outliers_features]
    
    # calculate quartiles
    lower_quartile = np.percentile(tx_outliers, low_bound, axis=0)
    upper_quartile = np.percentile(tx_outliers, high_bound, axis=0)
    
    # calculate index to be removed
    remove_index = np.argwhere((tx_outliers < lower_quartile) | (tx_outliers > upper_quartile))[:, 0]
    
    return_tx = np.delete(tx, remove_index, axis=0)
    return_y = np.delete(y, remove_index, axis=0)
    
    return (return_y, return_tx)

"""split the dataset based on the split ratio."""
def split_data(x, y, ratio, seed=1):
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te