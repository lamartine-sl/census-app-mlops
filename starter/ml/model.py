"""
Training and Scoring Module

Author : Lamartine Santana
Date : September 2021
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from .data import process_data

def train_model(X_train, y_train, model_params):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model_params: dict
        Model hyperparameters
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: CatBoostClassifier, X: np.array):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : CatBoostClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def slice_metrics_perfomance(
        df: pd.DataFrame,
        feature: str,
        model,
        encoder:OneHotEncoder,
        binarizer:LabelBinarizer):
    """
    Computes model metrics based on data slices
    Inputs
    ------
    df : pd.DataFrame
         Dataframe containing the cleaned data
    category : str
         Dataframe column to slice
    rf_model: 
         Random forest model used to perform prediction
    encoder: OneHotEncoder
         Trained OneHotEncoder
    binarizer: LabelBinarizer
        Trained LabelBinarizer
     Returns
     -------
     predictions : dict
          Dictionary containing the predictions for each category feature
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    
    predictions = {}
    for feat in df[feature].unique():
        df_temp = df[df[feature] == feat]
        X, y, _, _ = process_data(
            df_temp, 
            categorical_features=cat_features, 
            label="salary", 
            training=False, 
            encoder = encoder, 
            lb = binarizer)

        predict_values = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, predict_values)

        predictions[feat] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'rows': len(df_temp)}
    return predictions