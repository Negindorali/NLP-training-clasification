# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project actually is

A **from-scratch** sentiment classifier in **pure Python — no third-party libraries at all** (not even numpy). Every NLP/ML step — text cleaning, tokenizing, stopword removal, stemming, vocabulary, count vectorization, TF-IDF, and Multinomial Naive Bayes — is hand-implemented in `src/` using only the standard library (`math`, `re`). The venv has numpy/pandas/sklearn installed but **nothing in the code imports them** — keep it that way (the project goal is to implement everything by hand).

Matrices are plain Python lists of lists (`X[i][j]`), not numpy arrays. Don't reintroduce numpy when editing.

`text-classification-project-requirements.md` is the original aspirational brief (asks for sklearn, multiple classifiers, notebooks, web UI, cross-validation, etc.). It does **not** describe the current implementation. Treat it as a wishlist, not a spec — most of it is unbuilt, and the `models/` and `results/` directories are empty.

## Running

```bash
source venv/bin/activate
python main.py             # full pipeline (run from the project root)
python main_no_tfidf.py    # simplified single-file version (also from the root)
```

There are no tests, no linter config, and no `requirements.txt`. There are **no runtime dependencies** — standard-library Python 3.12 only.

### Entry points — two parallel versions

- `main.py` — the full pipeline, an **interactive terminal app** (REPL): trains on startup, then classifies any sentence you type, showing Positive/Negative/Neutral with a confidence bar. Commands: `:eval` (full test-set metrics + confusion matrix), `:words` (top informative words), `:threshold <n>` (live-tune the Neutral cutoff), `:help`, `:quit`. ANSI colors auto-disable when output isn't a TTY.
- `main_no_tfidf.py` — a deliberately **simple, self-contained beginner version** of the same task: raw word counts → Naive Bayes, everything in one file, no `src/` imports, no TF-IDF/stemming/stopwords. Heavily commented for teaching. Don't "improve" it by adding the pipeline machinery back — its simplicity is the point.
- `sentiment_notebook.ipynb` / `sentiment_notebook_no_tfidf.ipynb` — Jupyter walkthroughs mirroring the two scripts (load → train → classify → evaluate → top words). Both must run from the project root. The first drives `SentimentPipeline` directly — not `main()` (the REPL's `input()` loop doesn't belong in a notebook); the second inlines the beginner code like `main_no_tfidf.py`. Launch with `jupyter lab` / `jupyter notebook` from the venv.
- `regexSample.py` — a separate, self-contained interactive demo (tiny `.`/`*` regex engine) with the same REPL style — not part of the classifier.

When changing classifier behavior, keep the script and its matching notebook consistent.

Imports use the `src.` package prefix (e.g. `from src.naive_bayes import MultinomialNB`), but there is **no `src/__init__.py`** — it works only as a namespace package, so `main.py` must be executed from the repo root, not from inside `src/`.

## Pipeline architecture

`src/pipeline.py` (`SentimentPipeline`) is the orchestrator — it owns all stages, exposes `train(texts, labels)`, `predict(texts)`, and `classify_text(text) -> (label, probs)`, and is what `main.py` drives. The stages, chained in this exact order, are:

1. `SentenceDatasetLoader` (`sentence_loader.py`) — reads `data/*_labelled.txt`, format `sentence\tlabel`, returns `(texts, labels)` with integer labels.
2. `TextCleaner.clean` (`text_cleaner.py`) — lowercases, strips URLs/emails, keeps `[a-z\s]` only.
3. `Tokenizer.tokenize` (`tokenizer.py`) — whitespace split.
4. `StopwordRemover.remove` (`stopwords.py`) — hardcoded ~15-word stoplist.
5. `Stemmer.stem_tokens` (`stemmer.py`) — naive suffix-stripping (`ing/ed/ly/es/s`).
6. `Vocabulary.build` (`vocabulary.py`) — builds `word_to_index`/`index_to_word`; **fit on training docs only**. `min_df` (pipeline default 2) drops words seen in fewer than that many training docs — single-occurrence words are noise that made the model confidently wrong on unseen domains.
7. `CountVectorizerCustom.transform` (`vectorizer.py`) — bag-of-words matrix as a list of lists; out-of-vocabulary words are silently dropped.
8. `TFIDF` (`tfidf.py`) — **opt-in only** (`SentimentPipeline(use_tfidf=True)`, default off): idf gives rare words the largest weights, which amplifies one-example noise into high-confidence wrong predictions, so NB gets raw counts by default.
9. `MultinomialNB` (`naive_bayes.py`) — Laplace-smoothed multinomial NB; `alpha` sets the smoothing strength (higher = rare words sway predictions less). `classify(x) -> (label, probs)` is the core: normalizes log-scores via log-sum-exp and applies the Neutral threshold; `predict` is a thin wrapper. `top_informative_words` ranks by log-ratio of `P(word|class)` vs the other classes (discriminative, not just frequent).
10. `metrics.py` — `accuracy`, and `confusion_matrix(y_true, y_pred)` which returns `(matrix, labels)` sized to the labels present.

### Labels and the Neutral class — important

The data only contains **0 = Negative** and **1 = Positive**; there are **no neutral examples** in any of the three files (confirmed: 500/500 each). Neutral therefore cannot be *learned*.

`MultinomialNB(neutral_threshold=...)` produces Neutral (`NEUTRAL_LABEL = 2`) as a **low-confidence prediction**: if the gap between the top two class probabilities is below the threshold, it returns 2. `main.py` sets `0.3` — tune it (0 disables Neutral; higher pushes more sentences into Neutral). `main_no_tfidf.py` does the same thing with a `neutral_gap=0.20` string-label version. Because the test set has no true neutrals, every Neutral prediction counts as wrong in overall accuracy, so both entry points also report accuracy over only the confident (non-Neutral) predictions.

### Evaluation design (intentional, not a bug)

Both entry points train on **imdb + amazon** and test on **yelp** as an unseen domain (cross-domain generalization), rather than a random split.

### Sharp edges to respect when editing

- `Vocabulary` must be built before vectorizing, and `TFIDF`/`Vocabulary` are fit on train data then only `transform`-ed on test to avoid leakage — preserve this.
- NB is fit on **raw integer counts** (as multinomial NB assumes). TF-IDF is kept only as an opt-in comparison (`use_tfidf=True`) — it was found to make cross-domain predictions confidently wrong (e.g. 90% Negative for "I was seated immediately" off a single training example) and must not silently become the default again.
- `LABEL_NAMES` in `main.py` is the single source of truth for label→name display (0 Negative, 1 Positive, 2 Neutral).
