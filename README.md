# Word2Vec: Pure NumPy Implementation

A dependency-free, pure-NumPy implementation of the Word2Vec continuous space language models. This repository provides from-scratch implementations of both the **Skip-Gram** and **Continuous Bag-of-Words (CBOW)** architectures, utilizing **Negative Sampling (SGNS)** to approximate the full softmax denominator.

This project is designed to explicitly demonstrate the forward pass, objective function formulation, and manual backpropagation of gradients without relying on automatic differentiation engines (e.g., PyTorch Autograd or TensorFlow GradientTape).

## Architectural Overview

The model maps a discrete vocabulary $V$ to a continuous, lower-dimensional embedding manifold $d \ll |V|$. It maintains two distinct weight matrices:
1. **$W_{in} \in \mathbb{R}^{|V| \times d}$**: The input embedding matrix (representations for the center/target words in Skip-Gram, or context words in CBOW).
2. **$W_{out} \in \mathbb{R}^{|V| \times d}$**: The output/context embedding matrix.

To avoid the $\mathcal{O}(|V|)$ computational bottleneck of updating the full softmax distribution per step, we utilize Negative Sampling. This casts the multiclass classification problem into a series of binary logistic regression tasks, discriminating between true context pairs drawn from the empirical distribution and "noise" pairs drawn from a modified unigram distribution.

### The Objective Function (SGNS)

For a given center word $w_c$ and a positive context word $w_o$, we sample $K$ negative words $w_{n_k} \sim P_n(w)$. The objective is to maximize the log-likelihood of the observed pairs while minimizing it for the negative samples. The local loss function $\mathcal{L}$ for a single step is defined as:

$$\mathcal{L} = -\log \sigma(v_c^\top v_o) - \sum_{k=1}^K \log \sigma(-v_c^\top v_{n_k})$$

Where:
* $v_c$ is the vector for the center word (from $W_{in}$).
* $v_o$ is the vector for the true context word (from $W_{out}$).
* $v_{n_k}$ are the vectors for the $K$ negative samples (from $W_{out}$).
* $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid activation function.

### Gradient Derivations

The model updates parameters via Stochastic Gradient Descent (SGD). The partial derivatives of $\mathcal{L}$ with respect to the embedding vectors are:

**Gradient w.r.t the positive context vector:**
$$\frac{\partial \mathcal{L}}{\partial v_o} = (\sigma(v_c^\top v_o) - 1) v_c$$

**Gradient w.r.t each negative context vector:**
$$\frac{\partial \mathcal{L}}{\partial v_{n_k}} = (1 - \sigma(-v_c^\top v_{n_k})) v_c$$

**Gradient w.r.t the center word vector:**
$$\frac{\partial \mathcal{L}}{\partial v_c} = (\sigma(v_c^\top v_o) - 1) v_o + \sum_{k=1}^K (1 - \sigma(-v_c^\top v_{n_k})) v_{n_k}$$

## Algorithmic Optimizations

To ensure convergence and meaningful semantic clustering, this implementation includes several standard Mikolov et al. heuristics:

1. **Subsampling of Frequent Words:** Highly frequent tokens (e.g., function words) are discarded with probability $P_{keep}(w_i) = \min(1, \sqrt{\frac{t}{f(w_i)}} + \frac{t}{f(w_i)})$, where $f(w_i)$ is the token's frequency and $t$ is a heuristic threshold (default $10^{-3}$).
2. **Smoothed Unigram Distribution:** Negative samples are drawn from a smoothed noise distribution $P_n(w) \propto U(w)^{3/4}$, which aggressively penalizes the sampling of extremely rare words.
3. **Dynamic Context Windows:** The active window size $C$ is uniformly sampled $c \sim \mathcal{U}(1, W)$ for each target word, naturally weighting proximal words more heavily than distal ones.
4. **Linear Learning Rate Decay:** The learning rate $\eta$ decays linearly toward $\eta_{min}$ as a function of processed tokens, stabilizing the embedding manifold during late-stage training.
5. **Gradient Clipping:** To prevent exploding gradients early in the training phase due to poor random initialization, gradients are hard-clipped to $[-5.0, 5.0]$.

## Quick Start & CLI Usage
### Basic Training
Train the Skip-Gram architecture on a built-in toy corpus to verify gradient updates:

```bash
python word2vec_numpy.py --mode skipgram
```

### Fetching Online Corpora via REST API
Construct training sets dynamically by fetching plain-text article summaries directly from the free Wikipedia API:


```bash
python word2vec_numpy.py --mode skipgram --wikipedia "Support vector machine" "Transformer (deep learning architecture)"
```
### Hyperparameter Tuning
Train a CBOW model by scraping a Wikipedia category, adjusting the embedding dimensionality, context window, negative samples, and learning rate:

```bash
python word2vec_numpy.py \
    --wiki-category "Machine learning algorithms" \
    --category-limit 20 \
    --mode cbow \
    --dim 128 \
    --window 8 \
    --neg 10 \
    --epochs 100 \
    --lr 0.05 \
    --probe "algorithm" "data" "model"
```

## Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | string | `skipgram` | Architecture to use: `skipgram` or `cbow`. |
| `--wikipedia` | string(s) | None | Fetch specific Wikipedia article titles (space-separated). |
| `--wiki-category` | string | None | Fetch articles belonging to this Wikipedia category. |
| `--category-limit` | int | `10` | Max number of articles to fetch if using `--wiki-category`. |
| `--dim` | int | `50` | Size of the embedding vectors (dimensionality). |
| `--window` | int | `5` | Maximum context window size. |
| `--neg` | int | `5` | Number of negative samples drawn per positive context word. |
| `--epochs` | int | `50` | Number of training epochs. |
| `--lr` | float | `0.025` | Initial learning rate. |
| `--min-count` | int | `2` | Words appearing fewer than this many times are ignored. |
| `--keep-stopwords` | flag | False | Pass to prevent filtering out common stop words. |
| `--probe` | string(s) | Auto | Words to evaluate at the end of training for nearest neighbors. |