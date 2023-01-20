"""Nodes for feature engineering pipeline."""

import warnings

import pandas as pd
from loguru import logger
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from loguru import logger


def train_test_split_node(
    normalized_tweets: pd.DataFrame,
    sentiment_col_name: str,
    text_col_name: str,
    train_test_split_params: dict,
) -> pd.DataFrame:
    """Split the data into train and test sets.

    Args:
        normalized_tweets (pd.DataFrame): Nomalized tweets.
        sentiment_col_name (str): Name of the sentiment column.
        text_col_name (str): Name of the text column.
        train_test_split_params (dict): Parameters for the train test split.

    Returns:
        X_train (pd.DataFrame): Training set.
        X_test (pd.DataFrame): Test set.
        y_train (pd.DataFrame): Training set labels.
        y_test (pd.DataFrame): Test set labels.

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(normalized_tweets, split_ratio=0.2, random_state=42)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_tweets[text_col_name],
        normalized_tweets[sentiment_col_name],
        **train_test_split_params,
    )
    return X_train, X_test, y_train, y_test


def vectorize_data_node(X_train, X_test, tfidf_params, text_col_name) -> pd.DataFrame:
    """Vectorize the data.

    Args:
        X_train (pd.DataFrame): Training set.
        X_test (pd.DataFrame): Test set.
        tfidf_params (dict): Parameters for the tfidf vectorizer.
        text_col_name (str): Name of the text column.


    Returns:
        vectorized_train (pd.DataFrame): Vectorized training set.
        vectorized_test (pd.DataFrame): Vectorized test set.
    """
    # convert ngram_range to tuple (yaml does not support tuples)
    tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorized_train = vectorizer.fit_transform(X_train[text_col_name])
    vectorized_test = vectorizer.transform(X_test[text_col_name])
    return vectorized_train, vectorized_test, vectorizer
