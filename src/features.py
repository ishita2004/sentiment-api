import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# ── CONCEPT: TF-IDF ────────────────────────────────────────
# TF  = count(word in doc) / total words in doc
# IDF = log(total docs / docs containing word)
# TF-IDF = TF × IDF
#
# "the"         → appears everywhere → low IDF → low weight (noise)
# "masterpiece" → appears rarely     → high IDF → high weight (signal)
#
# N-GRAMS capture word sequences:
#   unigrams only:  ["not", "good"]    ← misses negation
#   + bigrams:      ["not good"]       ← captures negation ✓


POSITIVE_WORDS = {
    "amazing", "excellent", "fantastic", "wonderful", "great", "love",
    "best", "perfect", "brilliant", "outstanding", "superb", "awesome",
    "incredible", "recommend", "happy", "enjoyed", "impressive",
}
NEGATIVE_WORDS = {
    "terrible", "horrible", "awful", "worst", "hate", "boring",
    "disappointing", "waste", "poor", "bad", "broken", "useless",
    "return", "refund", "never", "cheap", "fake", "avoid", "regret",
}


class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    12 hand-crafted features that TF-IDF cannot capture:
    text length, exclamation marks, caps ratio, lexicon matches, etc.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(text) for text in X], dtype=np.float32)

    def _extract(self, text):
        words = text.lower().split()
        n = max(len(words), 1)
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        return [
            len(text),                                          # 0  text length
            n,                                                  # 1  word count
            np.mean([len(w) for w in words]) if words else 0,  # 2  avg word len
            text.count("!"),                                    # 3  exclamation
            text.count("?"),                                    # 4  question marks
            sum(c.isupper() for c in text) / len(text) if text else 0,  # 5  caps ratio
            pos,                                                # 6  positive word count
            neg,                                                # 7  negative word count
            (pos - neg) / (pos + neg + 1),                     # 8  sentiment ratio
            sum(1 for w in words if w in {"not","no","never"}), # 9  negation count
            text.count("..."),                                  # 10 ellipsis (often negative)
            sum(c.isdigit() for c in text) / len(text) if text else 0,  # 11 digit ratio
        ]


class SentimentFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Combines TF-IDF (sparse) + CustomFeatures (dense) horizontally.
    Final shape: (N, vocab_size + 12)
    """

    def __init__(self, ngram_range=(1, 2), max_features=30_000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,      # log(1+tf) — dampens high-freq terms
            token_pattern=r"\w{2,}",
        )
        self.custom = CustomFeatureExtractor()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        tfidf_mat = self.tfidf.transform(X)               # sparse (N, vocab)
        custom_mat = csr_matrix(self.custom.transform(X)) # sparse (N, 12)
        return hstack([tfidf_mat, custom_mat])             # sparse (N, vocab+12)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    texts = [
        "This is AMAZING! Best product ever!!!",
        "Terrible waste of money. Never buying again.",
        "It was okay, nothing special really...",
    ]
    fp = SentimentFeaturePipeline(max_features=500)
    mat = fp.fit_transform(texts)
    print(f"Feature matrix shape: {mat.shape}")

    custom = CustomFeatureExtractor().transform(texts)
    names = ["text_len","word_count","avg_word_len","exclamation","questions",
             "caps_ratio","pos_words","neg_words","sentiment_ratio",
             "negation_count","ellipsis","digit_ratio"]
    print("\nCustom features for 'AMAZING! Best product ever!!!':")
    for name, val in zip(names, custom[0]):
        print(f"  {name:20s}: {val:.3f}")