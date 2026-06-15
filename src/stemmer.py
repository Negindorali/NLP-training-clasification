class Stemmer:

    def stem(self, word):

        suffixes = ["ing","ed","ly","es","s"]

        for s in suffixes:

            if word.endswith(s) and len(word) > len(s)+2:
                return word[:-len(s)]

        return word

    def stem_tokens(self, tokens):

        return [self.stem(w) for w in tokens]
