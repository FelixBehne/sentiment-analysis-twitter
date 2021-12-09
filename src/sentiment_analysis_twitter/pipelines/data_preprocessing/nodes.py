"""Nodes for data preprocessing pipeline."""
import pandas as pd
from fast_ml.utilities import reduce_memory_usage
from loguru import logger

from .utils import abbr_dict, basic_cleaner, normalize, slang_dict


def basic_cleanup(tweets: pd.DataFrame) -> pd.DataFrame:
    """Removes urls, quotations, re-tweets, punctuations, repeated vowels (e.g. haha),
    non-ascii characters, hashtags and mentions.

    Args:
        tweets (pd.DataFrame): Raw training data.

    Returns:
        pd.DataFrame: Cleaned up tweets.
    """
    # deduplicate tweets
    tweets = tweets.drop_duplicates(keep="first")

    # drop nans
    tweets = tweets.dropna()

    # enforce string type
    tweets["text"] = tweets["text"].astype("str")

    # Create new df
    intermediate = pd.DataFrame()

    # apply basic cleaner to tweets
    intermediate["text"] = tweets["text"].map(lambda tweet: basic_cleaner(tweet))

    # Add target variable to intermediate ds
    intermediate["target"] = tweets.target

    return reduce_memory_usage(intermediate)


def normalize_tweets(
    tweets: pd.DataFrame, lemmatization: bool, spellcheck: bool
) -> pd.DataFrame:
    """Normalize tweets.

    Args:
        tweets (pd.DataFrame): Tweets that have been cleanup regarding the
        use of characters and punctuations.
        lemmatization (bool): Determines wether a lemmatization should be applied.
        spellcheck (bool): Determines wether a spellcheck should be applied.

    Returns:
        pd.DataFrame: Tweets with cleaned up language.
    """
    # replace nan, abbreviations and slang
    primary_tweets = (
        tweets.dropna().replace(abbr_dict, regex=True).replace(slang_dict, regex=True)
    )
    # remove quotations
    primary_tweets = primary_tweets.replace('"', "").replace("'", "")

    # enforce string type
    primary_tweets["text"] = primary_tweets["text"].astype("str")

    # # apply basic cleaner to tweets
    if lemmatization:
        logger.info("Applying lemmatization.")
    if spellcheck:
        logger.info("Applying spellcheck.")
    primary_tweets["text"] = primary_tweets["text"].map(
        lambda tweet: normalize(tweet, lemmatization, spellcheck)
    )
    return reduce_memory_usage(primary_tweets)
