"""Tiny dependency-free text helpers used by the agent (term extraction, counting)."""
import re

STOPWORDS = set("""
a an the is are was were be been am do does did can could would should will there this that these those
it its they them what which who whom whose where when why how many much any some of in on at to from by
with without for and or but not no yes image images picture pictures photo photos photograph scene visible
see seen show shown shows look looks looking appear appears contain contains include includes have has had
having near around about above below behind front side left right top bottom middle background foreground
tell list describe identify detect find locate count number total please also other another each every all
both few several one two three four five six seven eight nine ten first second likely possible potential
safety hazard hazards danger dangers dangerous unsafe issue issues concern concerns present object objects
thing things item items red blue green yellow white black gray grey orange brown pink purple day night
daytime nighttime outside inside outdoors indoors wearing wear wears standing walking sitting holding doing
happening going big small large tall little very more most less area part kind type color colour colors
colours here out up down over under between next anything something everything nothing
""".split())

IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child",
             "mice": "mouse", "feet": "foot", "buses": "bus", "glasses": "glasses"}
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_COUNT_RE = re.compile(
    r"how many\s+(.+?)(?:\s+(?:are|is|do|does|can|could|were|was|have|has|appear|visible|in|on|at|near|there)\b|\?|$)"
)


def singularize(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses", "zes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str):
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def extract_terms(text: str, known_phrases=(), max_terms: int = 6):
    """Return likely object nouns in `text` (multi-word known phrases first)."""
    t = text.lower()
    terms = []
    for phrase in sorted(known_phrases, key=len, reverse=True):
        if " " in phrase and phrase in t:
            terms.append(phrase)
            t = t.replace(phrase, " ")
    for tok in tokenize(t):
        if tok in STOPWORDS or len(tok) < 3:
            continue
        s = singularize(tok)
        if s in STOPWORDS or len(s) < 3:
            continue
        terms.append(s)
    return list(dict.fromkeys(terms))[:max_terms]


def count_targets(question: str, known_phrases=()):
    """For 'how many X ...' questions return the object terms to count, else []."""
    m = _COUNT_RE.search(question.lower().strip())
    return extract_terms(m.group(1), known_phrases, max_terms=3) if m else []


def is_yes_no(answer: str) -> bool:
    return answer.strip().lower().rstrip(".!") in {"yes", "no"}


def parse_number(text: str):
    """Extract an integer from a short answer such as '3' or 'two'."""
    m = re.search(r"\d+", text)
    if m:
        return int(m.group())
    for tok in tokenize(text):
        if tok in _NUMBER_WORDS:
            return _NUMBER_WORDS[tok]
    return None


def prompt_for(label: str) -> str:
    label = label.strip()
    if label.lower().startswith(("a ", "an ", "the ")):
        return f"a photo of {label}"
    return f"a photo of a {label}"
