"""
Sentiment classification WITHOUT TF-IDF  (simple beginner version).
===================================================================

The goal here is the SIMPLEST possible version:

    just count words  ->  Naive Bayes  ->  Positive / Negative / Neutral

No TF-IDF, no vocabulary class, no vectorizer, no stemmer. Everything is in
this one file using plain Python so it is easy to read and explain.

How Naive Bayes works (in plain words):
  1. Look at all the POSITIVE sentences and count how often each word appears.
     Do the same for the NEGATIVE sentences.
  2. For a new sentence, ask: "are these words more common in positive or in
     negative sentences?"
  3. Whichever side wins is the answer. If it's basically a tie, say Neutral.

Run it from the project root:

    python main_no_tfidf.py
"""

import math


# ---------------------------------------------------------------------------
# Step 1: load the data
# ---------------------------------------------------------------------------
# The files look like:   some sentence here<TAB>1
# where 1 = positive and 0 = negative.

def load_file(path):
    sentences = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentence, label = line.split("\t")
            sentences.append(sentence)
            labels.append(int(label))
    return sentences, labels


# ---------------------------------------------------------------------------
# Step 2: turn a sentence into a list of simple words
# ---------------------------------------------------------------------------
# Lowercase it and keep only letters, then split on spaces. That's it.

def to_words(sentence):
    clean = ""
    for ch in sentence.lower():
        clean += ch if ch.isalpha() else " "
    return clean.split()


# ---------------------------------------------------------------------------
# Step 3: train Naive Bayes (just counting)
# ---------------------------------------------------------------------------

def train(sentences, labels):
    # word_counts[0] = how many times each word appeared in NEGATIVE sentences
    # word_counts[1] = the same for POSITIVE sentences
    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}     # total words seen in each class
    doc_counts = {0: 0, 1: 0}      # how many sentences in each class
    vocabulary = set()             # every word we have ever seen

    for sentence, label in zip(sentences, labels):
        doc_counts[label] += 1
        for word in to_words(sentence):
            vocabulary.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    return word_counts, total_words, doc_counts, vocabulary


# ---------------------------------------------------------------------------
# Step 4: classify a new sentence
# ---------------------------------------------------------------------------

def classify(sentence, model, neutral_gap=0.20):
    word_counts, total_words, doc_counts, vocabulary = model
    vocab_size = len(vocabulary)
    total_docs = doc_counts[0] + doc_counts[1]

    # We work with log-probabilities so the numbers don't get too tiny.
    scores = {}
    for label in (0, 1):
        # start with how common this class is overall
        score = math.log(doc_counts[label] / total_docs)
        for word in to_words(sentence):
            # P(word | class) with add-1 (Laplace) smoothing so a brand new
            # word never makes the whole probability zero.
            count = word_counts[label].get(word, 0)
            prob = (count + 1) / (total_words[label] + vocab_size)
            score += math.log(prob)
        scores[label] = score

    # Turn the two log-scores into normal probabilities that add up to 1.
    biggest = max(scores.values())
    exp0 = math.exp(scores[0] - biggest)
    exp1 = math.exp(scores[1] - biggest)
    total = exp0 + exp1
    p_neg = exp0 / total
    p_pos = exp1 / total

    # If positive and negative are too close, we are not sure -> Neutral.
    if abs(p_pos - p_neg) < neutral_gap:
        answer = "Neutral"
    elif p_pos > p_neg:
        answer = "Positive"
    else:
        answer = "Negative"

    return answer, p_pos, p_neg


# ---------------------------------------------------------------------------
# Step 5: measure accuracy on a test set
# ---------------------------------------------------------------------------

def evaluate(sentences, labels, model):
    correct = 0
    neutral = 0
    for sentence, true_label in zip(sentences, labels):
        answer, _, _ = classify(sentence, model)
        if answer == "Neutral":
            neutral += 1
        elif (answer == "Positive" and true_label == 1) or \
             (answer == "Negative" and true_label == 0):
            correct += 1
    return correct, neutral, len(sentences)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    print()
    print("Sentiment Classifier  -  WITHOUT TF-IDF (just Naive Bayes)")
    print("=" * 58)

    # Train on imdb + amazon, test on yelp (a different, unseen set).
    s1, l1 = load_file("data/imdb_labelled.txt")
    s2, l2 = load_file("data/amazon_cells_labelled.txt")
    train_sentences = s1 + s2
    train_labels = l1 + l2
    test_sentences, test_labels = load_file("data/yelp_labelled.txt")

    print(f"Training on {len(train_sentences)} sentences (imdb + amazon)")
    print(f"Testing  on {len(test_sentences)} sentences (yelp)")

    # Train (= count words).
    model = train(train_sentences, train_labels)

    # Check how well it does on the test set.
    correct, neutral, total = evaluate(test_sentences, test_labels, model)
    print()
    print(f"Correct (Positive/Negative) : {correct} / {total}  "
          f"= {correct / total * 100:.1f}%")
    print(f"Answered Neutral (not sure) : {neutral} / {total}")
    print()

    # Let the user type their own sentences.
    print("Type a sentence (or just press Enter to quit).")
    print()
    while True:
        try:
            text = input("sentence> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if text == "":
            print("bye!")
            break

        answer, p_pos, p_neg = classify(text, model)
        print(f"  -> {answer}")
        print(f"     Positive {p_pos * 100:5.1f}%   Negative {p_neg * 100:5.1f}%")
        print()


if __name__ == "__main__":
    main()
