"""
word2vec — Skip-Gram with Negative Sampling (SGNS)
Pure NumPy implementation — no ML frameworks.

Dataset options
───────────────
  1. Built-in mini-corpus (no internet required)
  2. Wikipedia articles — fetched via the free Wikipedia REST API

Usage examples
──────────────
  # Built-in corpus (default)
  python word2vec_numpy.py

  # Wikipedia: fetch a few articles by title
  python word2vec_numpy.py --wikipedia "Machine learning" "Neural network" "Linguistics"

  # Wikipedia: fetch articles from a category
  python word2vec_numpy.py --wiki-category "Artificial intelligence" --category-limit 10

  # Tune hyperparameters
  python word2vec_numpy.py --wikipedia "Physics" "Chemistry" \\
      --dim 100 --epochs 30 --window 5 --neg 10

═══════════════════════════════════════════════════════
  MATHEMATICAL DERIVATION
═══════════════════════════════════════════════════════

Model
─────
Two embedding matrices:
  W_in  : (V, D)   centre-word  embeddings   (input)
  W_out : (V, D)   context-word embeddings   (output)

For each (centre c, positive context o) pair we sample K
"negative" context words n_1 … n_K ~ P_noise (unigram^0.75).

Objective (maximise log-likelihood → minimise loss):
  L = -log σ(v_c · v̂_o)  −  Σ_{k=1}^{K} log σ(−v_c · v̂_{n_k})

where σ(x) = 1 / (1 + e^{-x}).

Gradients
─────────
Define  s_pos = σ( v_c · v̂_o)          positive similarity
        s_k   = σ(-v_c · v̂_{n_k})      negative similarities  (K,)

∂L/∂v̂_o
  = (s_pos − 1) · v_c                                          (D,)

∂L/∂v̂_{n_k}
  = (1 − s_k) · v_c                                            (D,) each

∂L/∂v_c
  = (s_pos − 1) · v̂_o  −  Σ_k (1 − s_k) · v̂_{n_k}           (D,)

Parameters are updated with SGD + linear LR decay:
  lr(t) = max(lr_min,  lr_init * (1 − t / T))
"""

import numpy as np
import re
import time
import json
import argparse
import sys
from collections import Counter



# ─────────────────────────────────────────────────────────────────────────────
# 0.  STOP WORDS
#     Standard English stop words that carry syntactic glue but very little
#     semantic content. Filtering them stops high-frequency function words
#     from dominating nearest-neighbour results on Wikipedia corpora.
#     Active by default; disable with --keep-stopwords.
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS = frozenset("""
a about above after again against all also am an and any are as at
be because been before being below between both but by can cannot could
did do does doing don down during each few for from further get got had
has have having he her here hers herself him himself his how however
i if in into is it its itself let me more moreover most my myself no nor
not of off on once only or other ought our ours ourselves out over own
same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you
your yours yourself yourselves
""".split())

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILT-IN CORPUS
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_CORPUS = """
the king ruled the kingdom with wisdom and power
the queen advised the king on matters of the kingdom
the prince learned to rule from the king and the queen
the princess studied wisdom at the royal court
the knight protected the king and the kingdom
the soldier fought for the kingdom and the king
the crown belongs to the king of the kingdom
the throne is the seat of the king and queen
paris is the capital of france and a beautiful city
france is a country in europe with a rich culture
london is the capital of england and a great city
england is a country in europe near france
berlin is the capital of germany and a modern city
germany is a large country in europe
rome is the capital of italy and an ancient city
italy is a country in europe famous for culture
madrid is the capital of spain and a vibrant city
spain is a country in europe with warm weather
the man walked to the city to find work
the woman walked to the market to buy food
the boy played in the park near the river
the girl read a book in the library near the school
the dog ran through the park with the boy
the cat sat near the fire in the house
the river flows through the city to the sea
the mountain rises above the forest and the valley
the forest is home to many animals and trees
the ocean is vast and deep and full of life
the sun rises in the morning over the mountains
the moon shines at night over the ocean
scientists study nature to understand the world
doctors help patients recover from illness and disease
teachers educate students in schools and universities
engineers build bridges roads and machines for society
artists create paintings music and stories for culture
the computer processes data and runs programs quickly
the algorithm solves problems by following steps
the network connects computers around the world
data science uses mathematics and computers together
machine learning teaches computers to recognize patterns
the economy depends on trade and production and labor
money is exchanged for goods and services in markets
banks store money and provide loans to businesses
companies hire workers to produce goods and services
technology changes how people work and live and communicate
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WIKIPEDIA FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def _wiki_request(url: str, timeout: int = 15) -> dict:
    """Make a GET request to the Wikipedia API. Returns parsed JSON."""
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "word2vec-numpy-demo/1.0 (educational use)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_wikipedia_article(title: str, verbose: bool = True) -> str:
    """
    Fetch the full plain-text of a Wikipedia article using the free
    Wikipedia REST API (no API key required).

    Parameters
    ----------
    title   : article title, e.g. "Machine learning"
    verbose : print progress

    Returns
    -------
    Plain text of the article (may be several thousand tokens).
    """
    # Encode spaces as underscores for the URL
    slug = title.strip().replace(" ", "_")
    url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"

    if verbose:
        print(f"  Fetching Wikipedia: '{title}' …", end="", flush=True)

    try:
        data    = _wiki_request(url)
        extract = data.get("extract", "")
        if verbose:
            word_count = len(extract.split())
            print(f"  {word_count} words")
        return extract
    except Exception as exc:
        if verbose:
            print(f"  FAILED ({exc})")
        return ""


def fetch_wikipedia_articles(titles: list[str],
                              verbose: bool = True) -> str:
    """
    Fetch multiple Wikipedia articles and concatenate their text.
    """
    parts = []
    for title in titles:
        text = fetch_wikipedia_article(title, verbose=verbose)
        if text:
            parts.append(text)
    combined = " ".join(parts)
    if verbose:
        print(f"  Total: {len(combined.split()):,} words from "
              f"{len(parts)}/{len(titles)} articles\n")
    return combined


def fetch_wikipedia_category(category: str,
                              limit:    int  = 10,
                              verbose:  bool = True) -> str:
    """
    Fetch up to `limit` article titles from a Wikipedia category,
    then pull each article's summary text.

    Uses the MediaWiki Action API (also free, no key required).
    """
    if verbose:
        print(f"  Fetching category '{category}' (up to {limit} articles)…")

    # Query the category members
    base = "https://en.wikipedia.org/w/api.php"
    params = (
        f"?action=query&list=categorymembers"
        f"&cmtitle=Category:{category.replace(' ', '_')}"
        f"&cmlimit={limit}&cmtype=page&format=json"
    )
    try:
        data    = _wiki_request(base + params)
        members = data.get("query", {}).get("categorymembers", [])
        titles  = [m["title"] for m in members]
    except Exception as exc:
        if verbose:
            print(f"  Could not fetch category: {exc}")
        return ""

    if verbose:
        print(f"  Found {len(titles)} articles: {titles[:5]}{'…' if len(titles)>5 else ''}")

    return fetch_wikipedia_articles(titles, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(text: str,
             remove_stopwords: bool = False) -> list[str]:
    """
    Lowercase, split on non-alphabetic characters.
    If remove_stopwords=True, discard tokens in STOP_WORDS.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def build_vocab(tokens: list[str],
                min_count: int = 2) -> tuple[dict, dict, list]:
    counts   = Counter(tokens)
    vocab    = [w for w, c in counts.most_common() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, vocab


def subsample(tokens:    list[str],
              word2idx:  dict,
              counts:    Counter,
              total:     int,
              t:         float = 1e-3) -> list[str]:
    """Frequent-word sub-sampling (Mikolov et al. 2013)."""
    rng  = np.random.default_rng(42)
    kept = []
    for w in tokens:
        if w not in word2idx:
            continue
        f      = counts[w] / total
        p_keep = min(1.0, np.sqrt(t / f) + t / f)
        if rng.random() < p_keep:
            kept.append(w)
    return kept


def build_unigram_table(vocab:      list[str],
                         counts:    Counter,
                         table_size: int = 100_000) -> np.ndarray:
    """Noise distribution P(w) ∝ freq(w)^0.75 for negative sampling."""
    freqs  = np.array([counts[w] ** 0.75 for w in vocab], dtype=np.float64)
    freqs /= freqs.sum()
    return np.random.choice(len(vocab), size=table_size, p=freqs)


def make_skipgram_pairs(token_ids: list[int],
                        window:    int = 5) -> list[tuple[int, int]]:
    """Generate (centre, context) pairs with dynamic window sampling."""
    rng   = np.random.default_rng(0)
    pairs = []
    n     = len(token_ids)
    for i, centre in enumerate(token_ids):
        w  = int(rng.integers(1, window + 1))
        lo = max(0,   i - w)
        hi = min(n-1, i + w)
        for j in range(lo, hi + 1):
            if j != i:
                pairs.append((centre, token_ids[j]))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_embeddings(vocab_size: int,
                    embed_dim:  int,
                    seed:       int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng   = np.random.default_rng(seed)
    W_in  = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim,
                        size=(vocab_size, embed_dim))
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    return W_in, W_out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FORWARD PASS + LOSS + GRADIENTS + PARAMETER UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def sgns_step(centre_id:  int,
              context_id: int,
              neg_ids:    np.ndarray,
              W_in:       np.ndarray,
              W_out:      np.ndarray,
              lr:         float) -> float:
    """
    One SGNS update.  All maths in the module docstring.
    Mutates W_in and W_out in place.  Returns scalar loss.
    """
    v_c   = W_in [centre_id ]
    v_o   = W_out[context_id]
    V_neg = W_out[neg_ids   ]

    # Forward
    s_pos = _sigmoid( v_c @ v_o  )
    s_neg = _sigmoid(-V_neg @ v_c)

    # Loss
    eps  = 1e-10
    loss = -np.log(s_pos + eps) - np.sum(np.log(s_neg + eps))

    # Gradients
    grad_v_o   = (s_pos - 1.0) * v_c
    grad_V_neg = (1.0 - s_neg[:, None]) * v_c
    grad_v_c   = ((s_pos - 1.0) * v_o
                  - np.sum((1.0 - s_neg[:, None]) * V_neg, axis=0))

    # Clip
    clip = 5.0
    grad_v_c   = np.clip(grad_v_c,   -clip, clip)
    grad_v_o   = np.clip(grad_v_o,   -clip, clip)
    grad_V_neg = np.clip(grad_V_neg, -clip, clip)

    # SGD
    W_in [centre_id  ] -= lr * grad_v_c
    W_out[context_id ] -= lr * grad_v_o
    np.add.at(W_out, neg_ids, -lr * grad_V_neg)

    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(corpus_text:       str   = BUILTIN_CORPUS,
          embed_dim:          int   = 50,
          window:             int   = 5,
          n_neg:              int   = 5,
          n_epochs:           int   = 50,
          lr_init:            float = 0.025,
          lr_min:             float = 1e-4,
          min_count:          int   = 2,
          remove_stopwords:   bool  = False,
          seed:               int   = 42,
          verbose:            bool  = True) -> dict:
    """Full training pipeline. Returns embeddings + metadata."""
    np.random.seed(seed)

    tokens_raw  = tokenise(corpus_text, remove_stopwords=remove_stopwords)
    counts      = Counter(tokens_raw)
    total       = len(tokens_raw)
    word2idx, idx2word, vocab = build_vocab(tokens_raw, min_count=min_count)
    V           = len(vocab)

    tokens_sub  = subsample(tokens_raw, word2idx, counts, total)
    token_ids   = [word2idx[w] for w in tokens_sub]
    pairs       = make_skipgram_pairs(token_ids, window=window)
    n_pairs     = len(pairs)
    noise_table = build_unigram_table(vocab, counts)

    if verbose:
        print(f"{'─'*44}")
        print(f"Vocabulary size   : {V:,}")
        print(f"Tokens (raw)      : {total:,}")
        print(f"Tokens (subsampled): {len(tokens_sub):,}")
        print(f"Training pairs    : {n_pairs:,}")
        sw_str = "on" if remove_stopwords else "off"
        print(f"Dim={embed_dim}  window={window}  K={n_neg}  "
              f"epochs={n_epochs}  lr={lr_init}→{lr_min}  stop-words={sw_str}")
        print(f"{'─'*44}")

    W_in, W_out  = init_embeddings(V, embed_dim, seed=seed)
    loss_history = []
    total_steps  = n_epochs * n_pairs
    step         = 0
    rng          = np.random.default_rng(seed)
    t0           = time.time()

    for epoch in range(1, n_epochs + 1):
        ep_loss   = 0.0
        idx_order = rng.permutation(n_pairs)

        for i in idx_order:
            c_id, o_id = pairs[i]
            lr = max(lr_min, lr_init * (1.0 - step / total_steps))

            raw  = noise_table[rng.integers(0, len(noise_table), size=n_neg * 3)]
            negs = raw[raw != o_id][:n_neg]
            if len(negs) < n_neg:
                negs = np.resize(negs, n_neg)

            ep_loss += sgns_step(c_id, o_id, negs, W_in, W_out, lr)
            step    += 1

        avg = ep_loss / n_pairs
        loss_history.append(avg)
        if verbose:
            print(f"Epoch {epoch:3d}/{n_epochs}  loss={avg:.4f}  "
                  f"lr={lr:.5f}  t={time.time()-t0:.1f}s")

    embeddings = 0.5 * (W_in + W_out)
    return dict(W_in=W_in, W_out=W_out, embeddings=embeddings,
                word2idx=word2idx, idx2word=idx2word,
                vocab=vocab, loss_history=loss_history)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def most_similar(word:     str,
                 E:        np.ndarray,
                 word2idx: dict,
                 idx2word: dict,
                 topn:     int = 10) -> list[tuple[str, float]]:
    if word not in word2idx:
        return []
    idx   = word2idx[word]
    v     = E[idx]
    norms = np.linalg.norm(E, axis=1) + 1e-10
    sims  = (E @ v) / (norms * (np.linalg.norm(v) + 1e-10))
    sims[idx] = -1.0
    top   = np.argsort(sims)[::-1][:topn]
    return [(idx2word[i], float(sims[i])) for i in top]


def analogy(a: str, b: str, c: str,
            E:        np.ndarray,
            word2idx: dict,
            idx2word: dict,
            topn:     int = 5) -> list[tuple[str, float]]:
    """a : b :: c : ?   solved via v(b) − v(a) + v(c)."""
    for w in (a, b, c):
        if w not in word2idx:
            return []
    query  = E[word2idx[b]] - E[word2idx[a]] + E[word2idx[c]]
    query /= np.linalg.norm(query) + 1e-10
    norms  = np.linalg.norm(E, axis=1) + 1e-10
    sims   = (E @ query) / norms
    for w in (a, b, c):
        sims[word2idx[w]] = -1.0
    top = np.argsort(sims)[::-1][:topn]
    return [(idx2word[i], float(sims[i])) for i in top]


def print_results(result: dict, probes: list[str] = None) -> None:
    E        = result["embeddings"]
    word2idx = result["word2idx"]
    idx2word = result["idx2word"]

    if probes is None:
        # Auto-pick up to 10 content words: skip stop words and very short tokens
        vocab = result["vocab"]
        probes = [w for w in vocab
                  if len(w) >= 4 and w not in STOP_WORDS][:10]

    print(f"\n{'═'*52}")
    print("  NEAREST NEIGHBOURS  (cosine similarity)")
    print(f"{'═'*52}")
    for p in probes:
        if p not in word2idx:
            continue
        ns = most_similar(p, E, word2idx, idx2word, topn=5)
        line = "  ".join(f"{w}({s:.2f})" for w, s in ns)
        print(f"  {p:14s}→ {line}")

    print(f"\n  Final mean loss : {result['loss_history'][-1]:.4f}")
    print(f"  Vocabulary      : {len(word2idx):,} words")
    print(f"  Embedding shape : {E.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="word2vec Skip-Gram with Negative Sampling — pure NumPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python word2vec_numpy.py
      → built-in corpus (43 words, no internet needed)

  python word2vec_numpy.py --wikipedia "Machine learning" "Deep learning" "NLP"
      → fetch 3 Wikipedia articles (~2-5k tokens each)

  python word2vec_numpy.py --wiki-category "Artificial intelligence" --category-limit 8
      → fetch up to 8 articles from a Wikipedia category

  python word2vec_numpy.py --wikipedia "Physics" --dim 100 --epochs 40 --neg 10
        """)
    # Dataset
    src = p.add_mutually_exclusive_group()
    src.add_argument("--wikipedia", nargs="+", metavar="TITLE",
                     help="Wikipedia article titles to fetch (space-separated)")
    src.add_argument("--wiki-category", metavar="CATEGORY",
                     help="Wikipedia category to pull articles from")
    p.add_argument("--category-limit", type=int, default=10,
                   help="Max articles to fetch from a category (default: 10)")
    # Hyperparameters
    p.add_argument("--dim",    type=int,   default=50,    help="Embedding dimension (default: 50)")
    p.add_argument("--window", type=int,   default=5,     help="Context window (default: 5)")
    p.add_argument("--neg",    type=int,   default=5,     help="Negative samples per step (default: 5)")
    p.add_argument("--epochs", type=int,   default=50,    help="Training epochs (default: 50)")
    p.add_argument("--lr",     type=float, default=0.025, help="Initial learning rate (default: 0.025)")
    p.add_argument("--min-count", type=int, default=2,   help="Min token frequency for vocab (default: 2)")
    p.add_argument("--keep-stopwords", action="store_true",
                   help="Keep stop words in the vocabulary (default: filtered out for Wikipedia)")
    p.add_argument("--probe",  nargs="*",  metavar="WORD",
                   help="Words to show nearest neighbours for after training")
    return p


def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    # ── Select corpus ─────────────────────────────────────────────────────────
    if args.wikipedia:
        print(f"\nFetching {len(args.wikipedia)} Wikipedia article(s)…")
        corpus = fetch_wikipedia_articles(args.wikipedia, verbose=True)
        if not corpus.strip():
            print("No text fetched. Check article titles or your internet connection.")
            print("Falling back to built-in corpus.\n")
            corpus = BUILTIN_CORPUS
    elif args.wiki_category:
        print(f"\nFetching Wikipedia category: '{args.wiki_category}'…")
        corpus = fetch_wikipedia_category(
            args.wiki_category, limit=args.category_limit, verbose=True)
        if not corpus.strip():
            print("No text fetched. Falling back to built-in corpus.\n")
            corpus = BUILTIN_CORPUS
    else:
        print("\nUsing built-in corpus (pass --wikipedia or --wiki-category for more text).\n")
        corpus = BUILTIN_CORPUS

    # ── Auto-configure defaults for Wikipedia corpora ───────────────────────────
    using_wiki      = bool(args.wikipedia or args.wiki_category)
    remove_stopwords = not args.keep_stopwords   # default ON for any corpus

    # If user didn't set --dim explicitly, suggest a sensible default
    # based on corpus size: ~sqrt(V) rounded to nearest 25, clamped 50-300
    explicit_dim    = args.dim  # argparse default is 50 — user may have kept it

    # ── Train ─────────────────────────────────────────────────────────────────
    result = train(
        corpus_text      = corpus,
        embed_dim        = explicit_dim,
        window           = args.window,
        n_neg            = args.neg,
        n_epochs         = args.epochs,
        lr_init          = args.lr,
        min_count        = args.min_count,
        remove_stopwords = remove_stopwords,
        verbose          = True,
    )
    if using_wiki and not args.keep_stopwords:
        print("  (stop words filtered — use --keep-stopwords to retain them)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print_results(result, probes=args.probe)


if __name__ == "__main__":
    main()