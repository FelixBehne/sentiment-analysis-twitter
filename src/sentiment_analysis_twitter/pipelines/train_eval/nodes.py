"""Nodes for the train_eval pipeline."""
from pprint import pprint

import pandas as pd
import scipy
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report


def train_model_node(
    X_train: scipy.sparse.csr_matrix, y_train: pd.Series, model: BaseEstimator
):
    """[summary]

    Args:
        X_train (scipy.sparse.csr_matrix): [description]
        y_train (pd.Series): [description]
        model (BaseEstimator): [description]

    Returns:
        [type]: [description]
    """
    logger.info(f"Fitting {type(model).__name__} with the following parameters:")
    pprint(vars(model))
    model.fit(X_train, y_train["target"])
    return model


def evaluate_mode_node(
    Classifier: BaseEstimator, X_test: scipy.sparse.csr_matrix, y_test: pd.Series
):
    """[summary]

    Args:
        Classifier (BaseEstimator): [description]
        X_test (scipy.sparse.csr_matrix): [description]
        y_test (pd.Series): [description]
    """
    print(classification_report(y_test.target, Classifier.predict(X_test)))
