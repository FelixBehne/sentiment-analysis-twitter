"""Module for creating the data preprocessing pipeline"""

from kedro.pipeline import Pipeline, node
from .nodes import basic_cleanup, normalize_tweets


def create_pipeline():
    """Create the pipeline for data preprocessing."""
    return Pipeline(
        [
            node(
                basic_cleanup,
                inputs="raw_tweets",
                outputs="cleaned_tweets",
                name="basic_cleanup",
            ),
            node(
                normalize_tweets,
                inputs="cleaned_tweets",
                outputs="normalized_tweets",
                name="normalize_tweets",
            ),
        ]
    )
