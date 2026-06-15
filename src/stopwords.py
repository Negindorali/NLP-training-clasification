class StopwordRemover:

    def __init__(self):

        self.stopwords = {
            "the","is","and","a","of","to","in","for","on",
            "with","this","that","it","as","was"
        }

    def remove(self, tokens):

        return [w for w in tokens if w not in self.stopwords]
