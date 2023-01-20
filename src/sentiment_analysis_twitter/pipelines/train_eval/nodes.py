"""Nodes for the train_eval pipeline."""
import sys

import pandas as pd
import scipy
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<cyan>[{time:MM/DD/YY HH:mm:ss}]</cyan> <blue>{level: <8}</blue> <red>{message}</red>",
    level="INFO",
    backtrace=True,
    diagnose=True,
)
logger.add(
    "logs/training_pipeline.log",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    rotation="10 MB",
)


def train_model_node(
    X_train: scipy.sparse.csr_matrix,
    y_train: pd.Series,
    model_params: dict,
):
    rf_model = RandomForestClassifier(**model_params)
    rf_model.fit(X_train, y_train.values.ravel())
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
    # log accuracy, precision, recall, f1-score
    y_pred = classifier.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
    logger.info(f"Recall: {recall_score(y_test, y_pred, zero_division=0)}")
    logger.info(f"F1-score: {f1_score(y_test, y_pred, zero_division=0)}")
