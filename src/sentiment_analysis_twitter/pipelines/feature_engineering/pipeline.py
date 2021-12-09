"""Module for creating the feature engineering pipeline"""

from kedro.pipeline import Pipeline, node
from .nodes import train_test_split_node, vectorize_data_node


def create_pipeline():
    """Creates the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=train_test_split_node,
                inputs=["cleaned_tweets", "params:test_size"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
            node(
                func=vectorize_data_node,
                inputs=["X_train", "X_test", "params:vectorizer"],
                outputs=["X_train_vectorized", "X_test_vectorized"],
                name="vectorize_data_node",
            ),
        ]
    )
