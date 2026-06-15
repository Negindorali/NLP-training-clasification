import os
import sys


class SimpleRegex:
    """A tiny regex engine (pure Python, no libraries).

    Supported syntax:
        .   any single character
        *   zero or more of the preceding character

    The matcher reports the text index where a match ends so callers can
    recover both the matched substring and its position.
    """

    def match_char(self, p, t):
        return p == '.' or p == t

    def match_here(self, pattern, text, p_i, t_i):
        """Try to match pattern[p_i:] starting at text[t_i].

        Returns the text index just past the match, or None if no match.
        """

        # Whole pattern consumed -> success, ending at the current text index.
        if p_i == len(pattern):
            return t_i

        # A '*' applies to the current pattern char.
        if p_i + 1 < len(pattern) and pattern[p_i + 1] == '*':
            return self.match_star(pattern[p_i], pattern, text, p_i + 2, t_i)

        # Ordinary single-character match.
        if t_i < len(text) and self.match_char(pattern[p_i], text[t_i]):
            return self.match_here(pattern, text, p_i + 1, t_i + 1)

        return None

    def match_star(self, char, pattern, text, p_i, t_i):
        """Greedy '*': consume as many `char` as possible, then backtrack.

        Returns the text index just past the match, or None.
        """

        # Count the longest run of `char` (or '.') available.
        count = 0
        while t_i + count < len(text) and self.match_char(char, text[t_i + count]):
            count += 1

        # Try the longest match first, backing off until the rest fits.
        for k in range(count, -1, -1):
            end = self.match_here(pattern, text, p_i, t_i + k)
            if end is not None:
                return end

        return None

    def finditer(self, pattern, text):
        """Yield (start, end) spans of non-overlapping, non-empty matches."""

        spans = []
        i = 0
        n = len(text)

        while i <= n:

            end = self.match_here(pattern, text, 0, i)

            if end is not None and end > i:
                spans.append((i, end))
                i = end          # continue after this match (no overlap)
            else:
                i += 1

        return spans

    def search(self, pattern, text):
        """Return the matched substrings (convenience wrapper around finditer)."""

        return [text[s:e] for s, e in self.finditer(pattern, text)]


# ----------------------------------------------------------------------
# Terminal UI
# ----------------------------------------------------------------------

class Color:
    """ANSI color codes, auto-disabled when output isn't a real terminal."""

    enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    @classmethod
    def wrap(cls, text, code):
        if not cls.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    @classmethod
    def title(cls, t):
        return cls.wrap(t, "1;36")      # bold cyan

    @classmethod
    def dim(cls, t):
        return cls.wrap(t, "2")          # dim

    @classmethod
    def prompt(cls, t):
        return cls.wrap(t, "1;33")       # bold yellow

    @classmethod
    def match(cls, t):
        return cls.wrap(t, "1;30;42")    # bold black on green

    @classmethod
    def good(cls, t):
        return cls.wrap(t, "1;32")       # bold green

    @classmethod
    def warn(cls, t):
        return cls.wrap(t, "1;31")       # bold red


RULE = "─" * 52
INDENT = "  "

EXAMPLES = [
    ("the cat sat on a mat", ".at",     "'.' matches any char, so c/m/s + 'at'"),
    ("color colour colouur", "colou*r", "'u*' matches zero or more u's"),
    ("aaab xaaabz b",        "a*b",     "'a*' is greedy, then needs a 'b'"),
    ("hello world",          ".",       "every single character matches '.'"),
]


def highlight(text, spans):
    """Return the text with matched spans colored."""

    if not spans:
        return text

    out = []
    cursor = 0
    for start, end in spans:
        out.append(text[cursor:start])
        out.append(Color.match(text[start:end]))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def caret_line(text, spans):
    """A line of '^' markers sitting under each matched span."""

    marks = [" "] * len(text)
    for start, end in spans:
        for i in range(start, min(end, len(text))):
            marks[i] = "^"
    return "".join(marks).rstrip()


def show_result(regex, text, pattern):
    spans = regex.finditer(pattern, text)

    print()
    print(INDENT + highlight(text, spans))

    carets = caret_line(text, spans)
    if carets:
        print(INDENT + Color.good(carets))

    if not spans:
        print()
        print(INDENT + Color.warn("No matches."))
        print()
        return

    print()
    print(INDENT + Color.good(f"{len(spans)} match{'es' if len(spans) != 1 else ''}:"))

    # results table
    rows = [(str(i + 1), f"[{s},{e})", text[s:e]) for i, (s, e) in enumerate(spans)]
    w_num = max(len("#"), max(len(r[0]) for r in rows))
    w_span = max(len("span"), max(len(r[1]) for r in rows))

    header = f"{'#':>{w_num}}  {'span':<{w_span}}  text"
    print(INDENT + Color.dim(header))
    for num, span, frag in rows:
        print(INDENT + f"{num:>{w_num}}  {span:<{w_span}}  {frag}")
    print()


def print_help():
    print()
    print(INDENT + Color.title("Supported pattern syntax"))
    print(INDENT + "  .   matches any single character")
    print(INDENT + "  *   matches zero or more of the preceding character")
    print()
    print(INDENT + Color.title("Commands"))
    print(INDENT + "  :text       enter a new text to search in")
    print(INDENT + "  :examples   run a few built-in examples")
    print(INDENT + "  :help       show this help")
    print(INDENT + "  :quit       exit (also Ctrl-D / Ctrl-C)")
    print()


def run_examples(regex):
    print()
    for text, pattern, note in EXAMPLES:
        print(INDENT + Color.dim(f"# {note}"))
        print(INDENT + f"text:    {text}")
        print(INDENT + f"pattern: {pattern}")
        show_result(regex, text, pattern)


def ask(label):
    """Prompt for one line; returns None on EOF/interrupt."""
    try:
        return input(Color.prompt(label))
    except (EOFError, KeyboardInterrupt):
        return None


def main():
    regex = SimpleRegex()

    print()
    print(INDENT + Color.title("Simple Regex Matcher"))
    print(INDENT + Color.dim("supports  .  and  *   ·   type :help or :quit"))
    print(INDENT + RULE)

    text = ask("text> ")
    if text is None:
        print("\nbye!")
        return

    while True:

        entry = ask("regex> ")

        if entry is None:
            print("\nbye!")
            return

        command = entry.strip().lower()

        if command in (":quit", ":q", ":exit"):
            print("bye!")
            return

        if command in (":help", ":h", "?"):
            print_help()
            continue

        if command == ":examples":
            run_examples(regex)
            continue

        if command == ":text":
            new_text = ask("text> ")
            if new_text is None:
                print("\nbye!")
                return
            text = new_text
            continue

        if entry == "":
            print(INDENT + Color.dim("(empty pattern — enter a pattern, or :help)"))
            continue

        show_result(regex, text, entry)


if __name__ == "__main__":
    main()
