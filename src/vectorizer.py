class CountVectorizerCustom:

    def __init__(self, vocabulary):

        self.vocab = vocabulary

    def transform(self, documents):

        size = self.vocab.size()
        matrix = []

        for doc in documents:

            row = [0] * size

            for word in doc:

                if word in self.vocab.word_to_index:

                    j = self.vocab.word_to_index[word]
                    row[j] += 1

            matrix.append(row)

        return matrix
