"""Class that stores a tokenizer, tagger and lemmatizer to process strings."""

from string import punctuation

from flair.data import Sentence
from nltk.corpus import stopwords, wordnet


class TextProcessor:
    def __init__(self, tokenizer, tagger, lemmatizer):
        self.tokenizer = tokenizer
        self.tagger = tagger
        self.lemmatizer = lemmatizer
        self.exclude = stopwords.words("english") + [c for c in punctuation]

    def tokenize(self, text, filter=True):
        """Given a string, return the corresponding tokens as a list.If
        filter is set to false, stopwords are removed."""
        tokens = [token.text for token in Sentence(text, use_tokenizer=self.tokenizer)]
        if filter is True:
            return self.filter(tokens)
        return tokens

    def lemmatize(self, text, filter=True):
        """Given a string lower-case and lemmatize the words. The first step is
        to tokenize the string. Then, the tags of the words are computed. Those
        words that are either adjectives, nouns, verbs and adverbs are lemmatized
        whereas the rest are left intact. All words are lower-cased at the end
        and returned as a list. If filter is set to true, stopwords are
        removed."""
        sentence = Sentence(text, use_tokenizer=self.tokenizer)
        self.tagger.predict(sentence)
        tag_dict = {
            "ADJ": wordnet.ADJ,
            "NOUN": wordnet.NOUN,
            "VERB": wordnet.VERB,
            "ADV": wordnet.ADV,
        }
        lemmas = []
        for token in sentence:
            if token.labels[0].value in tag_dict:
                lemmas.append(
                    self.lemmatizer.lemmatize(
                        token.text.lower(), tag_dict[token.labels[0].value]
                    )
                )
            else:
                lemmas.append(token.text.lower())
        if filter is True:
            return self.filter(lemmas)
        return lemmas

    def filter(self, words):
        """Given a list of words, filter out punctuation signs and stopwords."""
        return [w for w in words if w not in self.exclude]
