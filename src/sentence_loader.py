from typing import List, Tuple


class SentenceDatasetLoader:

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> Tuple[List[str], List[int]]:

        sentences = []
        labels = []

        with open(self.filepath, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                sentence, label = line.split("\t")

                sentences.append(sentence)
                labels.append(int(label))

        return sentences, labels
