"""Nodes for the train_eval pipeline."""
from pprint import pprint

import pandas as pd
import scipy
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def train_model_node(
    X_train: scipy.sparse.csr_matrix,
    y_train: pd.Series,
    sentiment_col_name: str,
    text_col_name: str,
    model_params: dict,
):
    """[summary]

    Args:
        X_train (scipy.sparse.csr_matrix): [description]
        y_train (pd.Series): [description]
        model (BaseEstimator): [description]

    Returns:
        [type]: [description]
    """
    rf_model = RandomForestClassifier(**model_params)
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_mode_node(
    classifier: BaseEstimator, X_test: scipy.sparse.csr_matrix, y_test: pd.Series
):
    """[summary]

    Args:
        Classifier (BaseEstimator): [description]
        X_test (scipy.sparse.csr_matrix): [description]
        y_test (pd.Series): [description]
    """
    print(classification_report(y_test.target, classifier.predict(X_test)))
