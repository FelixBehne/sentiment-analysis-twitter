"""Nodes for data preprocessing pipeline."""
import pandas as pd
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

    # apply basic cleaner to tweets
    intermediate_tweets = pd.DataFrame()["text"] = tweets["text"].map(
        lambda tweet: basic_cleaner(tweet)
    )
    return intermediate_tweets


def normalize_tweets(
    tweets: pd.DataFrame, stopword_removal: bool, lemmatization: bool, spellcheck: bool
) -> pd.DataFrame:
    """Normalize tweets.

    Args:
        tweets (pd.DataFrame): Tweets that have been cleanup regarding the
        use of characters and punctuations.
        stopword_removal (bool): Determines wether stopwords should be removed or not.
        lemmatization (bool): Determines wether a lemmatization should be applied.
        spellcheck (bool): Determines wether a spellcheck should be applied.

    Returns:
        pd.DataFrame: Tweets with cleaned up language.
    """
    # replace nan, abbreviations and slang
    primary_tweets = (
        tweets.dropna().replace(abbr_dict, regex=True).replace(slang_dict, regex=True)
    )

    # enforce string type
    primary_tweets["text"] = primary_tweets["text"].astype("str")

    # # apply basic cleaner to tweets
    if stopword_removal:
        logger.info("Applying stopwords removal.")
    if lemmatization:
        logger.info("Applying lemmatization.")
    if spellcheck:
        logger.info("Applying spellcheck.")
    primary_tweets["text"] = primary_tweets["text"].map(
        lambda tweet: normalize(tweet, stopword_removal, lemmatization, spellcheck)
    )
    return primary_tweets
