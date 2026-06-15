from src.text_cleaner import TextCleaner
from src.tokenizer import Tokenizer
from src.stopwords import StopwordRemover
from src.stemmer import Stemmer
from src.vocabulary import Vocabulary
from src.vectorizer import CountVectorizerCustom
from src.tfidf import TFIDF
from src.naive_bayes import MultinomialNB


class SentimentPipeline:
    """End-to-end sentiment pipeline: clean -> tokenize -> stopwords -> stem
    -> bag-of-words -> TF-IDF -> Multinomial Naive Bayes.

    Wraps the individual stages so the UI can train once and then classify
    arbitrary sentences without re-wiring the steps each time.
    """

    def __init__(self, neutral_threshold=0.3):
        self.cleaner = TextCleaner()
        self.tokenizer = Tokenizer()
        self.stopwords = StopwordRemover()
        self.stemmer = Stemmer()

        self.vocab = Vocabulary()
        self.vectorizer = None
        self.tfidf = TFIDF()
        self.model = MultinomialNB(neutral_threshold=neutral_threshold)

    # -- preprocessing -------------------------------------------------

    def preprocess(self, texts):
        processed = []
        for text in texts:
            tokens = self.cleaner.clean(text)
            tokens = self.tokenizer.tokenize(tokens)
            tokens = self.stopwords.remove(tokens)
            tokens = self.stemmer.stem_tokens(tokens)
            processed.append(tokens)
        return processed

    def _features(self, texts):
        processed = self.preprocess(texts)
        counts = self.vectorizer.transform(processed)
        return self.tfidf.transform(counts)

    # -- training / inference -----------------------------------------

    def train(self, texts, labels):
        processed = self.preprocess(texts)

        self.vocab.build(processed)
        self.vectorizer = CountVectorizerCustom(self.vocab)

        counts = self.vectorizer.transform(processed)
        features = self.tfidf.fit_transform(counts)

        self.model.fit(features, labels)

    def predict(self, texts):
        return self.model.predict(self._features(texts))

    def classify_text(self, text):
        """Return (label, probs) for a single raw sentence."""
        features = self._features([text])
        return self.model.classify(features[0])

    # -- knobs ---------------------------------------------------------

    @property
    def neutral_threshold(self):
        return self.model.neutral_threshold

    @neutral_threshold.setter
    def neutral_threshold(self, value):
        self.model.neutral_threshold = value
