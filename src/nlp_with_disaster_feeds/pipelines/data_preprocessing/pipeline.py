"""Module for creating the data preprocessing pipeline"""

from kedro.pipeline import Pipeline, node
from .nodes import basic_cleanup, normalize_tweets


def create_pipeline():
    "Instantiates the data preprocessing pipeline"
    return Pipeline(
        [
            node(
                basic_cleanup,
                inputs="raw_tweets",
                outputs="intermediate_tweets",
                name="basic_cleanup",
            ),
            node(
                normalize_tweets,
                inputs=[
                    "intermediate_tweets",
                    "params:stopword_removal",
                    "params:lemmatization",
                    "params:spellcheck",
                ],
                outputs="cleaned_tweets",
                name="normalize_tweets",
            ),
        ]
    )
