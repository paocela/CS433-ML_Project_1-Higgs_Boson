import numpy as np

def preprocess_data(data, prediction, low, high, outlier_remove):
    data = substitute_nan_with_mean(data)
    data = log_transform(data)
    x, mean_x, std_x = standardize(data)
    if(outlier_remove == True):
        prediction, x = remove_outliers(prediction, x, low, high)
    return prediction, x

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
def substitute_nan_with_mean(x):
    x[x==-999] = np.nan
    avg_per_column = np.nanmean(x, axis=0)
    index_to_subst = np.where(np.isnan(x))
    x[index_to_subst] = np.take(avg_per_column, index_to_subst[1])
    return x

"""Substitute outliers using quantile ranges with the median """
def remove_outliers(y, x, low_bound, high_bound):
    # index columns with outliers visible from histograms
    index_outliers_features = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 13, 16, 19, 21, 23, 26, 27, 28, 29]
    
    # consider only related columns
    x_outliers = x[:, index_outliers_features]
    
    # calculate quartiles
    lower_quartile = np.percentile(x_outliers, low_bound, axis=0)
    upper_quartile = np.percentile(x_outliers, high_bound, axis=0)
    
    # calculate index to be removed
    remove_index = np.argwhere((x_outliers < lower_quartile) | (x_outliers > upper_quartile))[:, 0]
    
    return_x = np.delete(x, remove_index, axis=0)
    return_y = np.delete(y, remove_index, axis=0)
    
    return (return_y, return_x)

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

"""Log transform all features which show right-skewness and have strictly positive values"""
def log_transform(x):
    index_right_skewed = [0, 2, 5, 9, 10, 13, 16, 19, 21, 23, 26]
    
    return_x = np.copy(x)
    return_x[:, index_right_skewed] = np.log(return_x[:, index_right_skewed])
    
    return return_x

def drop_column(data):
    index_col = []
    return_data = np.delete(data, index_col, axis=1)
    return return_data

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly