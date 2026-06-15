class Vocabulary:

    def __init__(self):

        self.word_to_index = {}
        self.index_to_word = {}

    def build(self, documents):

        index = 0

        for doc in documents:

            for word in doc:

                if word not in self.word_to_index:

                    self.word_to_index[word] = index
                    self.index_to_word[index] = word

                    index += 1

    def size(self):

        return len(self.word_to_index)
