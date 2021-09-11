"""
This script training and compute metrics and inference model

Author: Lamartine Santana
Date: September 2021
"""
import os
import yaml
import logging
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, slice_metrics_perfomance
import pandas as pd
import numpy as np

CWD = os.getcwd()

logging.basicConfig(
    filename=os.path.join(
        CWD,
        'logs',
        'model.log'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    level=logging.INFO,
)

SLICE_LOGGER = logging.getLogger('slice_metrics')
SLICE_LOGGER.setLevel(logging.INFO)
SLICE_LOGGER.addHandler(
    logging.FileHandler(
        filename=os.path.join(
            CWD,
            'logs',
            'slice_output.txt'),
        mode='w',
    ))

with open(os.path.join(CWD, "starter", 'params.yaml'), 'r', encoding="UTF-8") as fp:
    CONFIG = yaml.safe_load(fp)

DATA_FILENAME = CONFIG['data']
DATA_DIR = os.path.join(
    CWD,
    'data'
)
DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

DATA = pd.read_csv(DATA_PATH)

# Add code to load in the data.
#data = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(
    DATA, 
    test_size=0.20,
    random_state=CONFIG['random_seed']
)

CAT_FEATURES = CONFIG['categorical_features']

X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=CAT_FEATURES, 
    label="salary", 
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=CAT_FEATURES, 
    label="salary", 
    training=False,
    encoder = encoder, 
    lb = lb
)

# Train and save a model.
model = train_model(X_train, y_train, CONFIG['train_params'])

# Predictions on the test data.
test_predicted_values = inference(model, X_test)

# Calculate the metrics.
precision, recall, fbeta = compute_model_metrics(y_test, test_predicted_values)


#Saving the encoder and the LabelBinarizer for being used in the API later
pd.to_pickle(model, "./model/model.pkl")
pd.to_pickle(encoder, "./model/encoder.pkl")
pd.to_pickle(lb, "./model/lb.pkl")
# Add code to load in the data, model and encoder
model = pd.read_pickle(r"./model/model.pkl")
encoder = pd.read_pickle(r"./model/encoder.pkl") 
binarizer = pd.read_pickle(r"./model/lb.pkl")

#slice_metrics_perfomance(DATA, "workclass", model, encoder, binarizer)

for elem in CAT_FEATURES:
    slice_metrics = slice_metrics_perfomance(DATA, elem, model, encoder, binarizer)

    SLICE_LOGGER.info("`%s` category", elem)
    for feature_val, metrics in slice_metrics.items():
        SLICE_LOGGER.info(
            "`%s` category -> precision: %s, recall: %s, fbeta: %s, numb.rows: %s -- %s.",
                    elem, metrics['precision'], metrics['recall'],
                    metrics['fbeta'], metrics['n_row'], feature_val)
    SLICE_LOGGER.info('\n')