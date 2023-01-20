"""Nodes for data preprocessing pipeline."""
import re

import pandas as pd
from nltk.stem import PorterStemmer

from .utils import abbr_dict, slang_dict


def basic_cleanup(tweets: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup of tweets.

    Args:
        tweets (pd.DataFrame): Raw tweets.

    Returns:
        pd.DataFrame: Cleaned up tweets.
    """
    # deduplicate tweets
    tweets = tweets.drop_duplicates(keep="first")

    # drop nans
    tweets = tweets.dropna()

    # Lowercase the tweets
    tweets["text"] = tweets["text"].apply(lambda x: x.lower())

    # Remove URLs
    tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"https?://\S+", "", x))

    # Remove hashtags
    tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"#\S+", "", x))

    # Remove mentions
    tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"@\S+", "", x))

    # Remove punctuation
    tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"[^\w\s]", "", x))

    return tweets


def normalize_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    """Normalizes tweets by stemming the text and removing repeated vowels, non-ascii characters, abbreviations, s
        and slang.

    Args:
        tweets (pd.DataFrame): Cleaned up tweets.

    Returns:
        pd.DataFrame: Normalized tweets.
    """
    # Remove abbreviations and slang
    primary_tweets = (
        tweets.dropna().replace(abbr_dict, regex=True).replace(slang_dict, regex=True)
    )

    # Remove repeated vowels
    primary_tweets["text"] = primary_tweets["text"].apply(
        lambda x: re.sub(r"([aeiou])\1+", r"\1", x)
    )

    # remove quotations
    primary_tweets = primary_tweets.replace('"', "").replace("'", "")

    # enforce string type
    primary_tweets["text"] = primary_tweets["text"].astype("str")

    # Stem the words
    stemmer = PorterStemmer()
    primary_tweets["text"] = primary_tweets["text"].apply(
        lambda x: " ".join([stemmer.stem(word) for word in x.split()])
    )

    # Remove leading and trailing spaces
    primary_tweets["text"] = primary_tweets["text"].apply(lambda x: x.strip())

    # drop nans, None, and empty strings
    primary_tweets = primary_tweets.dropna()
    primary_tweets = primary_tweets[primary_tweets["text"] != "None"]
    primary_tweets = primary_tweets[primary_tweets["text"] != ""]
    primary_tweets = primary_tweets[primary_tweets["text"] != " "]
    primary_tweets = primary_tweets[primary_tweets["text"] != "np.nan"]

    # drop duplicates
    primary_tweets = primary_tweets.drop_duplicates(keep="first")

    return primary_tweets
