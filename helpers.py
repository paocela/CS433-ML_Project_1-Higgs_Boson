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
def substitute_outliers(tx, low_bound, high_bound):
    return_tx = np.empty((tx.shape[1], tx.shape[0]))
    for index_row, row in enumerate(tx.T):
        median_row = np.median(row)
        a = np.array(row)
        upper_quartile = np.percentile(a, high_bound)
        lower_quartile = np.percentile(a, low_bound)
        
        for index, y in enumerate(a):
            if y < lower_quartile or y > upper_quartile:
                a[index] = median_row
                
        return_tx[index_row] = a
    return return_tx.T