"""Unittests for the nodes in the data preprocessing pipeline."""

from sentiment_analysis_twitter.pipelines.data_preprocessing.nodes import (
    basic_cleanup,
    normalize_tweets
)
import pandas as pd
import pytest

# def test_basic_cleanup(tweets):
#     """Test the basic_cleanup node."""
#     cleaned_tweets = basic_cleanup(tweets, "text")
#     assert cleaned_tweets.shape == (4, 1)
#     assert cleaned_tweets["text"].tolist() == [
#         "this is a
