class Vocabulary:

    def __init__(self, min_df=1):

        # A word must appear in at least min_df documents to enter the
        # vocabulary. Words seen in a single document carry noise
        # (one review's quirk), not reusable sentiment signal.
        self.min_df = min_df
        self.word_to_index = {}
        self.index_to_word = {}

    def build(self, documents):

        doc_freq = {}

        for doc in documents:

            for word in set(doc):

                doc_freq[word] = doc_freq.get(word, 0) + 1

        index = 0

        for doc in documents:

            for word in doc:

                if word not in self.word_to_index and doc_freq[word] >= self.min_df:

                    self.word_to_index[word] = index
                    self.index_to_word[index] = word

                    index += 1

    def size(self):

        return len(self.word_to_index)
