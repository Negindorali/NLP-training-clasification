import os
import sys
import time

from src.sentence_loader import SentenceDatasetLoader
from src.pipeline import SentimentPipeline
from src.naive_bayes import NEUTRAL_LABEL
from src.metrics import accuracy, confusion_matrix, precision_recall_f1


# 0 = Negative, 1 = Positive (the only labels in the data),
# 2 = Neutral (never in the data; only ever a low-confidence prediction).
LABEL_NAMES = {0: "Negative", 1: "Positive", NEUTRAL_LABEL: "Neutral"}

INDENT = "  "
RULE = "─" * 56


# ----------------------------------------------------------------------
# Styling (ANSI, auto-disabled when output isn't a terminal)
# ----------------------------------------------------------------------

class Color:
    enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    @classmethod
    def wrap(cls, text, code):
        return text if not cls.enabled else f"\033[{code}m{text}\033[0m"

    @classmethod
    def title(cls, t): return cls.wrap(t, "1;36")
    @classmethod
    def dim(cls, t): return cls.wrap(t, "2")
    @classmethod
    def prompt(cls, t): return cls.wrap(t, "1;33")
    @classmethod
    def pos(cls, t): return cls.wrap(t, "1;30;42")   # black on green
    @classmethod
    def neg(cls, t): return cls.wrap(t, "1;37;41")   # white on red
    @classmethod
    def neu(cls, t): return cls.wrap(t, "1;30;43")   # black on yellow
    @classmethod
    def green(cls, t): return cls.wrap(t, "32")
    @classmethod
    def red(cls, t): return cls.wrap(t, "31")


def label_badge(label):
    name = " " + LABEL_NAMES.get(label, str(label)) + " "
    if label == 1:
        return Color.pos(name)
    if label == 0:
        return Color.neg(name)
    return Color.neu(name)


def bar(frac, width=24):
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_data():
    """Train on imdb + amazon, test on yelp as an unseen domain."""
    imdb = SentenceDatasetLoader("data/imdb_labelled.txt").load()
    amazon = SentenceDatasetLoader("data/amazon_cells_labelled.txt").load()
    yelp = SentenceDatasetLoader("data/yelp_labelled.txt").load()

    train_texts = imdb[0] + amazon[0]
    train_labels = imdb[1] + amazon[1]
    test_texts, test_labels = yelp

    return train_texts, train_labels, test_texts, test_labels


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------

def show_prediction(pipeline, text):
    label, probs = pipeline.classify_text(text)
    top = max(probs.values())

    print()
    print(INDENT + "Prediction: " + label_badge(label) +
          Color.dim(f"   confidence {top * 100:.0f}%"))
    print()

    for c in sorted(probs):
        name = LABEL_NAMES.get(c, str(c))
        fill = bar(probs[c])
        fill = Color.green(fill) if c == 1 else Color.red(fill)
        print(INDENT + f"  {name:<9} {fill} {probs[c] * 100:5.1f}%")

    if label == NEUTRAL_LABEL:
        print()
        print(INDENT + Color.dim("  → the two classes are too close, so this is "
                                 "flagged Neutral"))
    print()


def show_evaluation(pipeline, test_texts, test_labels):
    print()
    print(INDENT + Color.title("Evaluation — Yelp (unseen domain)"))
    print(INDENT + RULE)

    predictions = pipeline.predict(test_texts)

    overall = accuracy(test_labels, predictions)

    confident = [(t, p) for t, p in zip(test_labels, predictions) if p != NEUTRAL_LABEL]
    neutral_count = len(predictions) - len(confident)

    print(INDENT + f"Overall accuracy           : {overall * 100:5.1f}%")
    if confident:
        c_acc = accuracy([t for t, _ in confident], [p for _, p in confident])
        print(INDENT + f"Accuracy (confident only)  : {c_acc * 100:5.1f}%   "
              + Color.dim("(Neutral predictions excluded)"))
    print(INDENT + f"Neutral predictions        : {neutral_count} / {len(predictions)}")

    # per-class precision / recall / F1 (neutral predictions count as wrong)
    per_class, macro = precision_recall_f1(test_labels, predictions)
    print()
    print(INDENT + Color.dim("Per-class metrics (Neutral predictions count as wrong)"))
    print(INDENT + f"{'':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>9}")
    for c in sorted(per_class):
        m = per_class[c]
        print(INDENT + f"{LABEL_NAMES.get(c, c):>10} "
              f"{m['precision'] * 100:9.1f}% {m['recall'] * 100:9.1f}% "
              f"{m['f1'] * 100:9.1f}% {m['support']:>9}")
    print(INDENT + f"{'macro avg':>10} "
          f"{macro['precision'] * 100:9.1f}% {macro['recall'] * 100:9.1f}% "
          f"{macro['f1'] * 100:9.1f}%")

    # confusion matrix
    matrix, labels = confusion_matrix(test_labels, predictions)
    print()
    print(INDENT + Color.dim("Confusion matrix (rows = true, cols = predicted)"))
    header = " " * 11 + "".join(f"{LABEL_NAMES.get(l, l):>10}" for l in labels)
    print(INDENT + header)
    for lbl, row in zip(labels, matrix):
        cells = "".join(f"{v:>10}" for v in row)
        print(INDENT + f"{LABEL_NAMES.get(lbl, lbl):>10} {cells}")
    print()


def show_top_words(pipeline, top_n=12):
    print()
    print(INDENT + Color.title(f"Top {top_n} most informative words per class"))
    print(INDENT + RULE)

    top_words = pipeline.model.top_informative_words(pipeline.vocab, top_n=top_n)

    columns = []
    for c in pipeline.model.classes:
        name = LABEL_NAMES.get(c, f"Class {c}")
        words = [w for w, _ in top_words[c]]
        columns.append((name, words))

    width = max(len(name) for name, _ in columns)
    width = max(width, max(len(w) for _, words in columns for w in words))

    headers = INDENT + "  ".join(f"{name:<{width}}" for name, _ in columns)
    print(Color.dim(headers))
    for i in range(top_n):
        row = "  ".join(f"{words[i]:<{width}}" if i < len(words) else " " * width
                        for _, words in columns)
        print(INDENT + row)
    print()


def show_mistakes(pipeline, test_texts, test_labels, top_n=10):
    """Show test sentences the model got wrong (Neutral doesn't count:
    it is 'not sure', not a wrong pick, so it is only tallied)."""

    print()
    print(INDENT + Color.title(f"Misclassified test sentences (first {top_n})"))
    print(INDENT + RULE)

    predictions = pipeline.predict(test_texts)

    wrong = [(text, t, p) for text, t, p in zip(test_texts, test_labels, predictions)
             if p != t and p != NEUTRAL_LABEL]
    neutral_count = sum(1 for p in predictions if p == NEUTRAL_LABEL)

    print(INDENT + Color.dim(f"{len(wrong)} confidently wrong, "
                             f"{neutral_count} flagged Neutral (not counted here)"))
    print()

    for i, (text, t, p) in enumerate(wrong[:top_n], 1):
        print(INDENT + f"{i:>2}. {text.strip()}")
        print(INDENT + "    true: " + label_badge(t) +
              "   predicted: " + label_badge(p))
        print()


def print_help(pipeline):
    print()
    print(INDENT + Color.title("How to use"))
    print(INDENT + "  Type any sentence and press Enter to classify it.")
    print()
    print(INDENT + Color.title("Commands"))
    print(INDENT + "  :eval              full evaluation on the Yelp test set")
    print(INDENT + "  :words             most informative words per class")
    print(INDENT + "  :mistakes [n]      show n misclassified test sentences (default 10)")
    print(INDENT + f"  :threshold <n>     set the Neutral threshold "
          f"(now {pipeline.neutral_threshold})")
    print(INDENT + "  :help              show this help")
    print(INDENT + "  :quit              exit (also Ctrl-D / Ctrl-C)")
    print()


# ----------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------

def ask(label):
    try:
        return input(Color.prompt(label))
    except (EOFError, KeyboardInterrupt):
        return None


def handle_threshold(pipeline, arg):
    try:
        value = float(arg)
    except ValueError:
        print(INDENT + Color.red("  threshold must be a number, e.g. :threshold 0.3"))
        return
    if not 0 <= value <= 1:
        print(INDENT + Color.red("  threshold should be between 0 and 1"))
        return
    pipeline.neutral_threshold = value
    print(INDENT + Color.dim(f"  neutral threshold set to {value} "
                             f"(0 disables Neutral; higher = more Neutral)"))


def main():
    print()
    print(INDENT + Color.title("Sentiment Classifier  ·  Naive Bayes (from scratch)"))
    print(INDENT + RULE)

    train_texts, train_labels, test_texts, test_labels = load_data()
    print(INDENT + Color.dim(f"training on {len(train_texts)} sentences "
                             f"(imdb + amazon), testing on {len(test_texts)} (yelp)"))

    sys.stdout.write(INDENT + "training model... ")
    sys.stdout.flush()
    start = time.time()

    pipeline = SentimentPipeline(neutral_threshold=0.3)
    pipeline.train(train_texts, train_labels)

    elapsed = time.time() - start
    print(Color.green("done") +
          Color.dim(f"  ({pipeline.vocab.size()} words, {elapsed:.1f}s)"))
    print(INDENT + Color.dim(f"count matrix: {len(train_texts)} x "
                             f"{pipeline.vocab.size()} (train)   ·   "
                             f"{len(test_texts)} x {pipeline.vocab.size()} (test)"))

    quick = accuracy(test_labels, pipeline.predict(test_texts))
    print(INDENT + Color.dim(f"test accuracy: {quick * 100:.1f}%   "
                             f"·   type a sentence, or :help"))
    print()

    while True:
        entry = ask("sentence> ")

        if entry is None:
            print("\nbye!")
            return

        stripped = entry.strip()
        command = stripped.lower()

        if command in (":quit", ":q", ":exit"):
            print("bye!")
            return
        if command in (":help", ":h", "?"):
            print_help(pipeline)
            continue
        if command == ":eval":
            show_evaluation(pipeline, test_texts, test_labels)
            continue
        if command == ":words":
            show_top_words(pipeline)
            continue
        if command.startswith(":mistakes"):
            parts = stripped.split(None, 1)
            try:
                count = int(parts[1]) if len(parts) > 1 else 10
            except ValueError:
                print(INDENT + Color.red("  usage: :mistakes 10"))
                continue
            show_mistakes(pipeline, test_texts, test_labels, top_n=count)
            continue
        if command.startswith(":threshold"):
            parts = stripped.split(None, 1)
            handle_threshold(pipeline, parts[1] if len(parts) > 1 else "")
            continue
        if stripped == "":
            continue

        show_prediction(pipeline, entry)


if __name__ == "__main__":
    main()
