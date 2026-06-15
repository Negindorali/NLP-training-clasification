import math

# The training data only contains negative (0) and positive (1) examples.
# A "neutral" class cannot be learned, so we emit it as a prediction when the
# model is not confident enough to choose between the two trained classes.
NEUTRAL_LABEL = 2


class MultinomialNB:

    def __init__(self, neutral_threshold=None):
        # If set (e.g. 0.3), predictions where the gap between the top two
        # class probabilities is smaller than this are reported as neutral.
        # If None, the model behaves as a plain binary classifier.
        self.neutral_threshold = neutral_threshold

    def fit(self, X, y):

        self.classes = sorted(set(y))
        self.class_probs = {}
        self.word_probs = {}

        n_docs = len(X)
        vocab_size = len(X[0]) if X else 0

        for c in self.classes:

            rows_c = [X[i] for i in range(n_docs) if y[i] == c]

            self.class_probs[c] = len(rows_c) / n_docs

            # Laplace (add-one) smoothing so unseen words don't zero out a class
            word_count = [1.0] * vocab_size

            for row in rows_c:
                for j in range(vocab_size):
                    if row[j]:
                        word_count[j] += row[j]

            total_words = sum(word_count)

            self.word_probs[c] = [wc / total_words for wc in word_count]

    def _class_probabilities(self, x):
        """Return normalized P(class | x) using log-scores + softmax."""

        log_scores = {}

        for c in self.classes:

            score = math.log(self.class_probs[c])

            for j in range(len(x)):
                if x[j]:
                    score += x[j] * math.log(self.word_probs[c][j])

            log_scores[c] = score

        # log-sum-exp normalization (subtract max for numerical stability)
        max_score = max(log_scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in log_scores.items()}
        total = sum(exp_scores.values())

        return {c: e / total for c, e in exp_scores.items()}

    def classify(self, x):
        """Classify one feature vector.

        Returns (label, probs) where probs maps each trained class to its
        probability. label is NEUTRAL_LABEL when the top two probabilities are
        too close (within neutral_threshold).
        """

        probs = self._class_probabilities(x)
        best = max(probs, key=probs.get)

        if self.neutral_threshold is not None:

            ordered = sorted(probs.values(), reverse=True)
            margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0

            if margin < self.neutral_threshold:
                return NEUTRAL_LABEL, probs

        return best, probs

    def predict(self, X):

        return [self.classify(x)[0] for x in X]

    def top_informative_words(self, vocab, top_n=10):
        """Words most distinctive of each class.

        Ranks by log P(word|class) - log(mean P(word|other classes)) so the
        result favors discriminative words instead of merely frequent ones.
        """

        top_words = {}

        for c in self.classes:

            probs = self.word_probs[c]
            others = [oc for oc in self.classes if oc != c]

            scores = []

            for i in range(len(probs)):

                word = vocab.index_to_word[i]

                if others:
                    other_mean = sum(self.word_probs[oc][i] for oc in others) / len(others)
                else:
                    other_mean = 1e-12

                score = math.log(probs[i]) - math.log(other_mean)
                scores.append((word, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            top_words[c] = scores[:top_n]

        return top_words
