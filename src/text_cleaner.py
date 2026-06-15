import re


class TextCleaner:

    def clean(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        text = re.sub(r"[^a-z\s]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text
