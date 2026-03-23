"""
word2vec — Skip-Gram & CBOW with Negative Sampling (SGNS)
Pure NumPy implementation — no ML frameworks.

Dataset options
───────────────
  1. Built-in mini-corpus (no internet required)
  2. Wikipedia articles — fetched via the free Wikipedia REST API

Usage examples
──────────────
  # Built-in corpus (Skip-gram by default)
  python word2vec_numpy.py

  # Run CBOW on the built-in corpus
  python word2vec_numpy.py --mode cbow

  # Wikipedia: fetch a few articles and use CBOW
  python word2vec_numpy.py --mode cbow --wikipedia "Machine learning" "Neural network"
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
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "word2vec-numpy-demo/1.0 (educational use)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_wikipedia_article(title: str, verbose: bool = True) -> str:
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


def fetch_wikipedia_articles(titles: list[str], verbose: bool = True) -> str:
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


def fetch_wikipedia_category(category: str, limit: int = 10, verbose: bool = True) -> str:
    if verbose:
        print(f"  Fetching category '{category}' (up to {limit} articles)…")

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

def tokenise(text: str, remove_stopwords: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def build_vocab(tokens: list[str], min_count: int = 2) -> tuple[dict, dict, list]:
    counts   = Counter(tokens)
    vocab    = [w for w, c in counts.most_common() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, vocab


def subsample(tokens: list[str], word2idx: dict, counts: Counter, total: int, t: float = 1e-3) -> list[str]:
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


def build_unigram_table(vocab: list[str], counts: Counter, table_size: int = 100_000) -> np.ndarray:
    freqs  = np.array([counts[w] ** 0.75 for w in vocab], dtype=np.float64)
    freqs /= freqs.sum()
    return np.random.choice(len(vocab), size=table_size, p=freqs)


def make_skipgram_pairs(token_ids: list[int], window: int = 5, seed: int = 0) -> list[tuple[int, int]]:
    rng   = np.random.default_rng(seed)
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

def init_embeddings(vocab_size: int, embed_dim: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng   = np.random.default_rng(seed)
    W_in  = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, size=(vocab_size, embed_dim))
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


def sgns_step(centre_id: int, context_id: int, neg_ids: np.ndarray, 
              W_in: np.ndarray, W_out: np.ndarray, lr: float) -> float:
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
    grad_v_c   = ((s_pos - 1.0) * v_o - np.sum((1.0 - s_neg[:, None]) * V_neg, axis=0))

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


def cbow_step(center_id: int, context_ids: list[int], neg_ids: np.ndarray,
              W_in: np.ndarray, W_out: np.ndarray, lr: float) -> float:
    if not context_ids:
        return 0.0
    
    # Context vectors (Input) and Center vector (Output/Target)
    v_contexts = W_in[context_ids]
    v_avg = np.mean(v_contexts, axis=0)
    
    v_target = W_out[center_id]
    V_neg = W_out[neg_ids]

    # Forward
    s_pos = _sigmoid(v_avg @ v_target)
    s_neg = _sigmoid(-V_neg @ v_avg)

    # Loss
    eps = 1e-10
    loss = -np.log(s_pos + eps) - np.sum(np.log(s_neg + eps))

    # Gradients
    grad_v_target = (s_pos - 1.0) * v_avg
    grad_V_neg = (1.0 - s_neg[:, None]) * v_avg
    grad_v_avg = ((s_pos - 1.0) * v_target - np.sum((1.0 - s_neg[:, None]) * V_neg, axis=0))

    # Clip
    clip = 5.0
    grad_v_target = np.clip(grad_v_target, -clip, clip)
    grad_V_neg    = np.clip(grad_V_neg,    -clip, clip)
    grad_v_avg    = np.clip(grad_v_avg,    -clip, clip)

    # Update Output Weights
    W_out[center_id] -= lr * grad_v_target
    np.add.at(W_out, neg_ids, -lr * grad_V_neg)

    # Update Input Weights (Divide gradient by number of context words)
    W_in[context_ids] -= (lr * grad_v_avg) / len(context_ids)

    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(corpus_text:       str   = BUILTIN_CORPUS,
          mode:               str   = "skipgram",
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
    
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    tokens_raw  = tokenise(corpus_text, remove_stopwords=remove_stopwords)
    counts      = Counter(tokens_raw)
    total       = len(tokens_raw)
    word2idx, idx2word, vocab = build_vocab(tokens_raw, min_count=min_count)
    V           = len(vocab)

    tokens_sub  = subsample(tokens_raw, word2idx, counts, total)
    token_ids   = [word2idx[w] for w in tokens_sub]

    # Data preparation based on mode
    if mode == "skipgram":
        pairs = make_skipgram_pairs(token_ids, window=window, seed=seed)
    else:
        pairs = []
        n_tokens = len(token_ids)
        for i, center in enumerate(token_ids):
            w = int(rng.integers(1, window + 1))
            lo = max(0, i - w)
            hi = min(n_tokens - 1, i + w)
            # Gather surrounding words, excluding the center word itself
            ctx = [token_ids[j] for j in range(lo, hi + 1) if j != i]
            if ctx: 
                pairs.append((center, ctx))

    n_pairs     = len(pairs)
    noise_table = build_unigram_table(vocab, counts)

    if verbose:
        print(f"{'─'*44}")
        print(f"Architecture      : {mode.upper()}")
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
    t0           = time.time()

    for epoch in range(1, n_epochs + 1):
        ep_loss   = 0.0
        idx_order = rng.permutation(n_pairs)

        for i in idx_order:
            c_id, o_id = pairs[i]
            lr = max(lr_min, lr_init * (1.0 - step / total_steps))

            raw  = noise_table[rng.integers(0, len(noise_table), size=n_neg * 3)]
            # Prevent sampling the positive context/target word as a negative
            exclude_id = o_id if mode == "skipgram" else c_id
            negs = raw[raw != exclude_id][:n_neg]
            
            if len(negs) < n_neg:
                negs = np.resize(negs, n_neg)

            if mode == "skipgram":
                # c_id is center, o_id is a single context word ID
                ep_loss += sgns_step(c_id, o_id, negs, W_in, W_out, lr)
            else:
                # c_id is target center, o_id is a list of context word IDs
                ep_loss += cbow_step(c_id, o_id, negs, W_in, W_out, lr)

            step += 1

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

def most_similar(word: str, E: np.ndarray, word2idx: dict, idx2word: dict, topn: int = 10) -> list[tuple[str, float]]:
    if word not in word2idx:
        return []
    idx   = word2idx[word]
    v     = E[idx]
    norms = np.linalg.norm(E, axis=1) + 1e-10
    sims  = (E @ v) / (norms * (np.linalg.norm(v) + 1e-10))
    sims[idx] = -1.0
    top   = np.argsort(sims)[::-1][:topn]
    return [(idx2word[i], float(sims[i])) for i in top]


def print_results(result: dict, probes: list[str] = None) -> None:
    E        = result["embeddings"]
    word2idx = result["word2idx"]
    idx2word = result["idx2word"]

    if probes is None:
        vocab = result["vocab"]
        probes = [w for w in vocab if len(w) >= 4 and w not in STOP_WORDS][:10]

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
        description="word2vec Skip-Gram & CBOW with Negative Sampling — pure NumPy",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Dataset
    src = p.add_mutually_exclusive_group()
    src.add_argument("--wikipedia", nargs="+", metavar="TITLE",
                     help="Wikipedia article titles to fetch (space-separated)")
    src.add_argument("--wiki-category", metavar="CATEGORY",
                     help="Wikipedia category to pull articles from")
    
    # Hyperparameters & Settings
    p.add_argument("--mode",   type=str,   choices=["skipgram", "cbow"], default="skipgram", help="Model architecture (default: skipgram)")
    p.add_argument("--category-limit", type=int, default=10, help="Max articles to fetch from a category (default: 10)")
    p.add_argument("--dim",    type=int,   default=50,    help="Embedding dimension (default: 50)")
    p.add_argument("--window", type=int,   default=5,     help="Context window (default: 5)")
    p.add_argument("--neg",    type=int,   default=5,     help="Negative samples per step (default: 5)")
    p.add_argument("--epochs", type=int,   default=50,    help="Training epochs (default: 50)")
    p.add_argument("--lr",     type=float, default=0.025, help="Initial learning rate (default: 0.025)")
    p.add_argument("--min-count", type=int, default=2,   help="Min token frequency for vocab (default: 2)")
    p.add_argument("--keep-stopwords", action="store_true", help="Keep stop words in the vocabulary")
    p.add_argument("--probe",  nargs="*",  metavar="WORD", help="Words to show nearest neighbours for after training")
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
        corpus = fetch_wikipedia_category(args.wiki_category, limit=args.category_limit, verbose=True)
        if not corpus.strip():
            print("No text fetched. Falling back to built-in corpus.\n")
            corpus = BUILTIN_CORPUS
    else:
        print("\nUsing built-in corpus (pass --wikipedia or --wiki-category for more text).\n")
        corpus = BUILTIN_CORPUS

    using_wiki       = bool(args.wikipedia or args.wiki_category)
    remove_stopwords = not args.keep_stopwords

    # ── Train ─────────────────────────────────────────────────────────────────
    result = train(
        corpus_text      = corpus,
        mode             = args.mode,
        embed_dim        = args.dim,
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