"""Integration tests for the feature engineering pipeline."""

import pytest
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline, node
from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment_analysis_twitter.pipelines.feature_engineering.nodes import (
    train_test_split_node,
    vectorize_data_node,
)
from sentiment_analysis_twitter.pipelines.feature_engineering.pipeline import (
    create_pipeline,
)


@pytest.fixture
def mocked_catalog(cleaned_tweets):
    catalog = DataCatalog()
    catalog.add("cleaned_tweets", MemoryDataSet(cleaned_tweets))
    catalog.add("params:sentiment_col_name", MemoryDataSet("sentiment"))
    catalog.add("params:text_col_name", MemoryDataSet("text"))
    catalog.add(
        "params:train_test_split_params",
        MemoryDataSet({"random_state": 42, "test_size": 0.25, "shuffle": True}),
    )
    catalog.add(
        "params:tfidf_params",
        MemoryDataSet(
            {
                "strip_accents": "ascii",
                "lowercase": True,
                "analyzer": "word",
                "stop_words": "english",
                "token_pattern": "(?u)\b\w\w+\b",
                "ngram_range": [1, 1],  # will be converted to tuple in node
                "min_df": 1,
                "max_features": None,
                "vocabulary": None,
                "binary": False,
                "norm": "l2",
                "use_idf": True,
                "smooth_idf": True,
                "sublinear_tf": False,
            }
        ),
    )
    return catalog


@pytest.fixture
def pipeline():
    return Pipeline(
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
                outputs=["vectorized_train", "vectorized_test"],
                name="vectorize_data_node",
            ),
        ]
    )


def test_pipeline(
    seq_runner,
    mocked_catalog,
    pipeline,
):
    # run the pipeline
    output = seq_runner.run(pipeline, mocked_catalog)

    # check the output
    assert output["vectorized_train"].shape == (3, 2)
    assert output["vectorized_test"].shape == (1, 3)


# def test_pipeline_with_mocked_nodes(
#     mocker,
#     seq_runner,
#     catalog,
#     cleaned_tweets,
#     pipeline,
# ):
#     mocker.patch(
#         "sentiment_analysis_twitter.pipelines.feature_engineering.nodes.train_test_split_node",
#         return_value=(
#             cleaned_tweets,
#             cleaned_tweets,
#             cleaned_tweets,
#             cleaned_tweets,
#         ),
#     )
#     mocker.patch(
#         "sentiment_analysis_twitter.pipelines.feature_engineering.nodes.vectorize_data_node",
#         return_value=(
#             cleaned_tweets,
#             cleaned_tweets,
#         ),
#     )
#     output = seq_runner.run(pipeline, catalog)

#     assert output["vectorized_train"].shape == (3, 3)
#     assert output["vectorized_test"].shape == (1, 3)


# def test_pipeline_defined_in_registry(
#     seq_runner,
#     catalog,
# ):

#     pipeline = create_pipeline()

#     # run the pipeline
#     output = seq_runner.run(pipeline, catalog)

#     # check the output
#     assert output["vectorized_train"].shape == (3, 3)
#     assert output["vectorized_test"].shape == (1, 3)


# def test_pipeline_with_mocked_catalog(
#     mocker,
#     seq_runner,
#     cleaned_tweets,
#     pipeline,
# ):
#     catalog = mocker.patch(
#         {
#             "cleaned_tweets": cleaned_tweets,
#             "params:split_ratio": 0.25,
#             "params:random_state": 42,
#             "params:vectorizer": TfidfVectorizer(),
#         },
#     )
#     output = seq_runner.run(pipeline, catalog)

#     assert output["vectorized_train"].shape == (3, 3)
#     assert output["vectorized_test"].shape == (1, 3)
#     assert output["X_train"].shape == (3, 3)
#     assert output["X_test"].shape == (1, 3)
#     assert output["y_train"].shape == (3, 3)
#     assert output["y_test"].shape == (1, 3)
