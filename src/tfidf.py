import math


class TFIDF:

    def fit(self, X):

        N = len(X)
        num_features = len(X[0]) if X else 0

        # document frequency: how many documents contain each word
        df = [0] * num_features

        for row in X:
            for j in range(num_features):
                if row[j] > 0:
                    df[j] += 1

        # smoothed inverse document frequency
        self.idf = [math.log((N + 1) / (df[j] + 1)) + 1 for j in range(num_features)]

    def transform(self, X):

        result = []

        for row in X:

            total = sum(row) + 1e-9  # avoid divide-by-zero on empty docs

            result.append([(row[j] / total) * self.idf[j] for j in range(len(row))])

        return result

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
