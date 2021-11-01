# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *

print("Loading training data...")

# Loading train data
DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
prediction, data, ids = load_csv_data(DATA_TRAIN_PATH)

print("Training data loaded")

# Divide train data into subgroups based on characteristic feature 22
prediction_0, prediction_1, prediction_2, data_0, data_1, data_2, ids_0, ids_1, ids_2 = categorize_data_feature_PRIjetnum(prediction, data, ids)

print("Subgroup training data created")

# percentile definition for outliers removal
low = 1
high = 99

# Training data cleaning and preprocessing
prediction_0, data_0 = preprocess_data(data_0, prediction_0, low, high, True, 0)
prediction_1, data_1 = preprocess_data(data_1, prediction_1, low, high, True, 1)
prediction_2, data_2 = preprocess_data(data_2, prediction_2, low, high, True, 2)

# Feature expansion
data_0 = build_poly(data_0, 2)
data_0 = np.delete(data_0, 0, axis=1)
data_1 = build_poly(data_1, 2)
data_1 = np.delete(data_1, 0, axis=1)
data_2 = build_poly(data_2, 2)
data_2 = np.delete(data_2, 0, axis=1)

# Model creation
y_0, tx_0 = build_model_data(prediction_0, data_0)
y_1, tx_1 = build_model_data(prediction_1, data_1)
y_2, tx_2 = build_model_data(prediction_2, data_2)

print("Training data cleanined, preprocessed and data model created")

# Logistic regression for all subgroups
max_iters = 500
initial_w = np.zeros(tx_0.shape[1])
gamma = 0.3
print("Logistic regression subgroup 0...")
logistic_w_0, logistic_loss_0 = logistic_regression(y_0, tx_0, initial_w, max_iters, gamma)

print("Logistic regression subgroup 1...")
initial_w = np.zeros(tx_1.shape[1])
logistic_w_1, logistic_loss_1 = logistic_regression(y_1, tx_1, initial_w, max_iters, gamma)

print("Logistic regression subgroup 2...")
initial_w = np.zeros(tx_2.shape[1]) 
logistic_w_2, logistic_loss_2 = logistic_regression(y_2, tx_2, initial_w, max_iters, gamma)

# Loading test data
DATA_TEST_PATH = 'data/test.csv'
prediction_te, data_te, ids_te = load_csv_data(DATA_TEST_PATH)

print("Test data loaded")

# # Divide test data into subgroups based on characteristic feature 22
prediction_0_te, prediction_1_te, prediction_2_te, data_0_te, data_1_te, data_2_te, ids_0_te, ids_1_te, ids_2_te = categorize_data_feature_PRIjetnum(prediction_te, data_te, ids_te)

print("Subgroup test data created")

# Test data cleaning and preprocessing (no outlier removal)
prediction_0_te, data_0_te = preprocess_data(data_0_te, prediction_0_te, low, high, False, 0)
prediction_1_te, data_1_te = preprocess_data(data_1_te, prediction_1_te, low, high, False, 1)
prediction_2_te, data_2_te = preprocess_data(data_2_te, prediction_2_te, low, high, False, 2)

# Feature expansion
data_0_te = build_poly(data_0_te, 2)
data_0_te = np.delete(data_0_te, 0, axis=1)
data_1_te = build_poly(data_1_te, 2)
data_1_te = np.delete(data_1_te, 0, axis=1)
data_2_te = build_poly(data_2_te, 2)
data_2_te = np.delete(data_2_te, 0, axis=1)

# Model creation
y_0_te, tx_0_te = build_model_data(prediction_0_te, data_0_te)
y_1_te, tx_1_te = build_model_data(prediction_1_te, data_1_te)
y_2_te, tx_2_te = build_model_data(prediction_2_te, data_2_te)

print("Test data cleanined, preprocessed and data model created")

print("Calculating predictions...")
# Prediction calculation
y_pred_0 = predict_labels(reg_logistic_w_0, tx_0_te)
y_pred_1 = predict_labels(reg_logistic_w_1, tx_1_te)
y_pred_2 = predict_labels(reg_logistic_w_2, tx_2_te)

# Join predictions from different subgroups together
y_pred_final = build_final_predictions(y_pred_0, y_pred_1, y_pred_2, prediction_te.shape[0], ids_0_te, ids_1_te, ids_2_te)

print("Creating final submission...")

OUTPUT_PATH = 'submission.csv'
create_csv_submission(ids_te, y_pred_final, OUTPUT_PATH)

print("Done!")