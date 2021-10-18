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
    avg = np.zeros([tx.shape[1]])
    for i in range(tx.shape[1]):
        sum = 0
        n = 0
        for j in range(tx.shape[0]):
            if tx[j,i] != -999:
                sum += tx[j,i]
                n += 1
        avg[i] = sum/n
        for j in range(tx.shape[0]):
            if tx[j,i] == -999:
                tx[j,i] = avg[i]
    return tx

"""Substitute outliers using quantile ranges with the median """
def substitute_outliers(tx, low_bound, high_bound):
    return_tx = np.array([])
    for row in tx:
        median_row = np.median(row)
        a = np.array(row)
        upper_quartile = np.percentile(a, high_bound)
        lower_quartile = np.percentile(a, low_bound)
        for index, y in enumerate(a):
            if y < lower_quartile and y > upper_quartile:
                a[index] = median_row
        np.append(return_tx, a)
    return return_tx