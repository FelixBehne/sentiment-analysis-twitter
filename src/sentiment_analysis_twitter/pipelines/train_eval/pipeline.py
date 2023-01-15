"""Creates the train eval pipeline."""

from kedro.pipeline import Pipeline, node
from .nodes import train_model_node, evaluate_mode_node


def create_pipeline():
    """Creates the train eval pipeline."""
    return Pipeline(
        [
            node(
                func=train_model_node,
                inputs=[
                    "X_train_vectorized",
                    "y_train",
                    "params:text_col_name",
                    "params:sentiment_col_name",
                    "params:model_params",
                ],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_mode_node,
                inputs=["classifier", "X_test_vectorized", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
