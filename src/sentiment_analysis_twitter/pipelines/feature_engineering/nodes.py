"""Nodes for feature engineering pipeline."""

import warnings

import pandas as pd
from loguru import logger
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def train_test_split_node(
    cleaned_tweets: pd.DataFrame,
    sentiment_col_name: str,
    text_col_name: str,
    train_test_split_params: dict,
) -> pd.DataFrame:
    """Split the data into train and test sets.

    Args:
        cleaned_tweets (pd.DataFrame): Cleaned tweets.
        sentiment_col_name (str): Name of the sentiment column.
        text_col_name (str): Name of the text column.
        train_test_split_params (dict): Parameters for the train test split.

    Returns:
        X_train (pd.DataFrame): Training set.
        X_test (pd.DataFrame): Test set.
        y_train (pd.DataFrame): Training set labels.
        y_test (pd.DataFrame): Test set labels.

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(cleaned_tweets, split_ratio=0.2, random_state=42)
    """
    # drop nans in text column if there are any
    if cleaned_tweets[text_col_name].isnull().values.any():
        cleaned_tweets_without_nan = cleaned_tweets.dropna(subset=[text_col_name])
        logger.info(
            f"Removed {len(cleaned_tweets) - len(cleaned_tweets_without_nan)} rows with NaN values in the text column."
        )
    else:
        cleaned_tweets_without_nan = cleaned_tweets

    # check if all required parameters are in params
    if not all(
        param in train_test_split_params
        for param in ["test_size", "random_state", "stratify"]
    ):
        raise ValueError(
            "The train_test_split_params dictionary must contain the following parameters: test_size(float), random_state(int) and stratify(bool)."
        )

    # check if stratify is true
    if "stratify" in train_test_split_params and train_test_split_params[
        "stratify"
    ] not in [None, False]:
        # set strafity parameter to the sentiment column
        train_test_split_params["stratify"] = cleaned_tweets_without_nan[
            sentiment_col_name
        ]

    else:
        # remove stratify from params
        train_test_split_params.pop("stratify")

    # try to split the data into train and test sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_tweets_without_nan[text_col_name],
            cleaned_tweets_without_nan[sentiment_col_name],
            **train_test_split_params,
        )
    # if stratify is not possible warn the user, log class distribution and fall back to no stratify
    except ValueError:
        warnings.warn(
            "Stratify is not possible. Falling back to no stratify. Please check the class distribution."
        )
        logger.info(
            f"Class distribution: {cleaned_tweets_without_nan[sentiment_col_name].value_counts()}"
        )
        train_test_split_params.pop("stratify")
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_tweets_without_nan[text_col_name],
            cleaned_tweets_without_nan[sentiment_col_name],
            **train_test_split_params,
        )

    # convert to dataframes
    X_train = pd.DataFrame(X_train, columns=[text_col_name])
    X_test = pd.DataFrame(X_test, columns=[text_col_name])
    y_train = pd.DataFrame(y_train, columns=[sentiment_col_name])
    y_test = pd.DataFrame(y_test, columns=[sentiment_col_name])

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

    # convert to dataframes
    vectorized_train = pd.DataFrame(
        vectorized_train.toarray(), columns=vectorizer.get_feature_names_out()
    )
    vectorized_test = pd.DataFrame(
        vectorized_test.toarray(), columns=vectorizer.get_feature_names_out()
    )

    return vectorized_train, vectorized_test, vectorizer
