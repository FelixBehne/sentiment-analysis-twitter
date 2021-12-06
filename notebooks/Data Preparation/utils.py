"""Define util functions for data preparation"""
import re
from typing import List

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

abbr_dict = {
    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",
    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",
    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",
    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
}


def basic_cleaner(tweet: str) -> str:
    """Basic tweet cleanup function.

    Args:
        tweet (str): Raw tweet.

    Returns:
        str: Cleaned tweet.
    """
    # remove urls
    clean_text = re.sub(
        r"pic.twitter.com\S+", " ", re.sub(r"(http|https):\S+", " ", tweet)
    )

    # remove quotations
    clean_text = clean_text.replace("\t", " ").replace("\n", " ")
    clean_text = re.sub(r'"', "'", clean_text)

    # remove punctuations
    clean_text = re.sub(
        r"(\.|,|:|;|\?|!|\)|\(|\-|\[|\]|\{|\}|\*|\||\<|\>|%|&|/|$|\+|@|#|\$|£|=|\^|~)",
        " ",
        clean_text,
    )
    # remove haha and variants
    clean_text = re.sub("hh+", "h", clean_text)
    clean_text = re.sub("aaa+", "a", clean_text)
    clean_text = re.sub(r"(ah){2,}|(ha){2,}", " laugh ", clean_text)

    # remove repeated vowels
    clean_text = re.sub("[a]{3,}", "aa", clean_text)
    clean_text = re.sub("[e]{3,}", "ee", clean_text)
    clean_text = re.sub("[i]{3,}", "ii", clean_text)
    clean_text = re.sub("[o]{3,}", "oo", clean_text)
    clean_text = re.sub("[u]{3,}", "uu", clean_text)

    # remove hashtags
    clean_text = re.sub(r"#\S+", " ", clean_text)

    # remove the @ (mentions)
    clean_text = re.sub(r"@\S+", " ", clean_text)

    # remove the RT
    clean_text = re.sub(r"(RT )", " ", clean_text)

    # remove non ascii
    tmp = ""
    for char in clean_text:
        if ord(char) < 128:
            tmp += char
    clean_text = tmp

    # remove redundant whitespaces
    clean_text = re.sub(" +", " ", clean_text)

    # strip messages
    clean_text = clean_text.strip()

    return clean_text


def normalize_tweets(tweet: str) -> List[str]:
    """[summary]

    Args:
        tweet (str): [description]

    Returns:
        str: [description]
    """
    # Fix word lengthening
    clean_text = re.compile(r"(.)\1{2,}").sub(r"\1\1", tweet)

    clean_text = clean_text.lower()
    clean_text = clean_text.strip()
    clean_text = re.sub(" +", " ", clean_text)

    # Word lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_text = lemmatizer.lemmatize(clean_text)

    # Replace abbreviations with common word
    for abbrev, _ in abbr_dict.items():
        clean_text = clean_text.replace(abbrev, abbr_dict[abbrev])

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(clean_text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    return filtered_sentence
