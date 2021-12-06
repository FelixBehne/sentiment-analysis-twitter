"""Define util functions for data preparation"""
import re
from typing import List, Union

import nltk
import preprocessor as p
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

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
    "shoulda": "should have",
    "ive": "i have",
    "im": "i am ",
    "its": "it is",
}

slang_dict = {
    "tnx": "thanks",
    "thx": "thanks",
    "kk": "ok",
    "k": "ok",
    "lulz": "lol",
    "sry": "sorry",
    "l8": "late",
    "w8": "wait",
    "m8": "mate",
    "gr8": "great",
    "plz": "please",
    "pls": "please",
    "2moro": "tomorrow",
    "somthin": "something",
    "srsly": "seriously",
    "sht": "shit",
    "imho": "imo",
    "omfg": "omg",
    "bros": "brother",
    "bro": "brother",
    "nope": "no",
}


def basic_cleaner(tweet: str) -> str:
    """Basic tweet cleanup function.

    Args:
        tweet (str): Raw tweet.

    Returns:
        str: Cleaned tweet.
    """
    # lowercase tweets
    clean_text = tweet.lower()

    # remove urls, hashtags, mentions and rts
    clean_text = p.clean(clean_text)

    # remove quotations
    clean_text = clean_text.replace('"', "")

    # remove punctuations
    clean_text = re.sub(
        r"(\.|,|:|;|\?|!|\)|\(|\-|\[|\]|\{|\}|\*|\||\<|\>|%|&|/|$|\+|@|#|\$|£|=|\^|~)",
        " ",
        clean_text,
    )
    # remove haha and variants
    clean_text = re.sub("hh+", "h", clean_text)
    clean_text = re.sub("aaa+", "a", clean_text)
    clean_text = re.sub(r"(ah){2,}|(ha){2,}", "laugh", clean_text)

    # remove repeated vowels
    clean_text = re.sub("[a]{3,}", "aa", clean_text)
    clean_text = re.sub("[e]{3,}", "ee", clean_text)
    clean_text = re.sub("[i]{3,}", "ii", clean_text)
    clean_text = re.sub("[o]{3,}", "oo", clean_text)
    clean_text = re.sub("[u]{3,}", "uu", clean_text)

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

    # remove digits
    clean_text = "".join([i for i in clean_text if not i.isdigit()])

    # remove paragraphs
    clean_text = clean_text.replace("\t", " ").replace("\n", " ").strip()

    # remove double spacing
    clean_text = " ".join(clean_text.split())

    # Fix word lengthening
    clean_text = re.compile(r"(.)\1{2,}").sub(r"\1\1", clean_text)

    return clean_text


def normalize(
    tweet: str, stopword_removal: bool, lemmatization: bool, spellcheck: bool
) -> Union[List[str], str]:
    """Normalizes tweets by applying lemmatization and a topword removal.

    Args:
        tweet (str): Tweets that should be normalized.
        stopword_removal (bool): Determines wether stopwords should be removed or not.
        lemmatization (bool): Determines wether a lemmatization should be applied.
        spellcheck (bool): Determines wether a spellcheck should be applied.


    Returns:
        Union[List[str], str]: Normalized tweets. Return type depends on
        wether stopwords are removed or not.
    """
    # Word lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        clean_text = lemmatizer.lemmatize(tweet)
    else:
        clean_text = tweet

    # Fix misspelled words
    if spellcheck:
        spell = SpellChecker()
        corrected_text = []
        misspelled_words = spell.unknown(clean_text.split())
        for word in clean_text.split():
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        clean_text = " ".join(corrected_text)

    # Remove stopwords
    if stopword_removal:
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(clean_text)
        clean_text = [w for w in word_tokens if not w.lower() in stop_words]

    return clean_text
