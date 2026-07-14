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
    -> bag-of-words -> Multinomial Naive Bayes.

    Wraps the individual stages so the UI can train once and then classify
    arbitrary sentences without re-wiring the steps each time.

    NB is fed raw integer counts, as the multinomial model assumes. TF-IDF
    weighting is kept as an opt-in (use_tfidf=True) for comparison, but it
    inflates the evidence of rare words — a word seen once in training gets
    the highest idf weight — which makes the model confidently wrong on
    unseen-domain text instead of uncertain.
    """

    def __init__(self, neutral_threshold=0.3, min_df=2, alpha=1.0,
                 use_tfidf=False):
        self.cleaner = TextCleaner()
        self.tokenizer = Tokenizer()
        self.stopwords = StopwordRemover()
        self.stemmer = Stemmer()

        self.vocab = Vocabulary(min_df=min_df)
        self.vectorizer = None
        self.use_tfidf = use_tfidf
        self.tfidf = TFIDF() if use_tfidf else None
        self.model = MultinomialNB(alpha=alpha,
                                   neutral_threshold=neutral_threshold)

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
        if self.use_tfidf:
            return self.tfidf.transform(counts)
        return counts

    # -- training / inference -----------------------------------------

    def train(self, texts, labels):
        processed = self.preprocess(texts)

        self.vocab.build(processed)
        self.vectorizer = CountVectorizerCustom(self.vocab)

        features = self.vectorizer.transform(processed)
        if self.use_tfidf:
            features = self.tfidf.fit_transform(features)

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
