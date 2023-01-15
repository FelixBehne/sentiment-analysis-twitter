"""Unit tests for the feature engineering nodes."""

from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment_analysis_twitter.pipelines.feature_engineering.nodes import (
    train_test_split_node,
    vectorize_data_node,
)
import pytest
import warnings


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_test_split_node(cleaned_tweets, conf_params):
    X_train, X_test, y_train, y_test = train_test_split_node(
        cleaned_tweets,
        "sentiment",
        "text",
        split_ratio=conf_params["split_ratio"],
        random_state=conf_params["random_state"],
        stratify=conf_params["stratify"],
    )
    assert X_train.shape == (3, 1)
    assert X_test.shape == (1, 1)
    assert y_train.shape == (3, 1)
    assert y_test.shape == (1, 1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_test_split_node_with_mocked_config(cleaned_tweets):
    with pytest.raises(
        ValueError,
    ):
        train_test_split_node(
            cleaned_tweets,
            "sentiment",
            "text",
            train_test_split_params={
                "test_size": 0.2,
                "random_state": 42,
            },
        )

    with pytest.warns(
        UserWarning,
        match="Stratify is not possible. Falling back to no stratify. Please check the class distribution.",
    ):
        X_train, X_test, y_train, y_test = train_test_split_node(
            cleaned_tweets,
            "sentiment",
            "text",
            train_test_split_params={
                "test_size": 0.2,
                "random_state": 42,
                "stratify": True,
            },
        )

        assert X_train.shape == (3, 1)
        assert X_test.shape == (1, 1)
        assert y_train.shape == (3, 1)
        assert y_test.shape == (1, 1)


def test_vectorize_data_node(X_train, X_test):
    # vectorizer is only tested with TfidfVectorizer because the parsing it from the config would
    # be a huge overhead and it is tested with the Integration test anyway
    vectorized_train, vectorized_test = vectorize_data_node(
        X_train,
        X_test,
        vectorizer=TfidfVectorizer(),
    )

    assert vectorized_train.shape == (3, 3)
    assert vectorized_test.shape == (1, 3)
