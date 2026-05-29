import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# ⚠️ KEY CONCEPT: Never remove negation words!
# "not good" → if you remove "not" → "good" → WRONG prediction
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "nothing", "hardly"}
CUSTOM_STOP_WORDS = _stop_words - NEGATION_WORDS  # keep negation!


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)           # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)    # remove URLs
    # expand contractions BEFORE lowercasing (preserve "not")
    text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can't", "cannot", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)    # keep ! and ? (emotion signals)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOP_WORDS or t in NEGATION_WORDS]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# sklearn-compatible transformer so it plugs into Pipeline
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # stateless, nothing to learn

    def transform(self, X):
        return [clean(text) for text in X]


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "This movie was <b>absolutely</b> NOT good. Don't waste your time!!!",
        "I can't believe how amazing this is — best purchase ever :)",
        "Check this out: http://spam.com — worst thing I've seen",
    ]
    pp = TextPreprocessor()
    for orig, cleaned in zip(samples, pp.transform(samples)):
        print(f"ORIG : {orig}")
        print(f"CLEAN: {cleaned}")
        print()