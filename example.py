""" Example of how to use the text processor.

Installation steps:

1. Install requirements: pip install -r requirements.txt
2. Install spacy model: python -m spacy download en_core_web_sm
3. Open a python shell and execute the following commands:
    a. import nltk
    b. nltk.download("english")
    c. nltk.download("omw-1.4")
"""


from flair.tokenization import SpacyTokenizer
from flair.models import SequenceTagger
from nltk.stem import WordNetLemmatizer

from processor import TextProcessor


if __name__ == "__main__":
    tokenizer = SpacyTokenizer("en_core_web_sm")
    lemmatizer = WordNetLemmatizer()
    tagger = SequenceTagger.load("upos-fast")
    processor = TextProcessor(tokenizer, tagger, lemmatizer)
    text = "This text is going to be processed; in different ways!"
    print(processor.tokenize(text, filter=False))
    print(processor.tokenize(text))
    print(processor.lemmatize(text, filter=False))
    print(processor.lemmatize(text))
