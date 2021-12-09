"""This module contains all the nodes for the Feature Engineering pipeline."""

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def train_test_split_node(
    cleaned_tweets: pd.DataFrame, test_size: float
) -> pd.DataFrame:
    """Train test split

    Args:
        cleaned_tweets (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    # dropna
    cleaned_tweets.dropna(subset=["text"], inplace=True)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_tweets.drop(columns="target"),
        cleaned_tweets["target"],
        test_size=test_size,
        random_state=44,
    )
    return X_train, X_test, y_train, y_test


def vectorize_data_node(
    X_train,
    X_test,
    vectorizer: TransformerMixin = TfidfVectorizer,
) -> pd.DataFrame:
    """Vectorizer.

    Args:
        X_train (pd.DataFrame): Train data.
        test (pd.DataFrame): Test data.
        vectorizer (TransformerMixin): Vectorizer that should be applied.
        Defaults to sklearn.feature_extraction.text.TfidfVectorizer.

    Returns:
        pd.DataFrame: [description]
    """

    # Transform data
    vectorized_train = vectorizer.fit_transform(X_train["text"])
    vectorized_test = vectorizer.transform(X_test["text"])

    return vectorized_train, vectorized_test
