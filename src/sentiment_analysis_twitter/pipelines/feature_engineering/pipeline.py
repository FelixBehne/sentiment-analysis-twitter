"""Define the feature engineering pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_test_split_node,
    vectorize_data_node,
)


def create_pipeline() -> Pipeline:
    """Create the feature engineering pipeline."""
    return pipeline(
        [
            node(
                func=train_test_split_node,
                inputs=[
                    "cleaned_tweets",
                    "params:sentiment_col_name",
                    "params:text_col_name",
                    "params:train_test_split_params",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
            node(
                func=vectorize_data_node,
                inputs=[
                    "X_train",
                    "X_test",
                    "params:tfidf_params",
                    "params:text_col_name",
                ],
                outputs=["X_train_vectorized", "X_test_vectorized", "vectorizer"],
                name="vectorize_data_node",
            ),
        ]
    )
