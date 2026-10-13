"""Transformer self-attention over mRNA sequences, implemented in PyTorch.

The mechanism
-------------
Self-attention lets every position of a sequence decide how much to attend to
every other position. Each position produces three vectors from the same input:

* **Q (Query)** - what this position is looking for.
* **K (Key)**   - what this position offers to be matched against.
* **V (Value)** - the content this position contributes once matched.

The computation, implemented step by step in :meth:`SelfAttention.forward`:

    Q = X Wq,  K = X Wk,  V = X Wv          linear projections, (seq_len, d_k)

    scores = Q K^T / sqrt(d_k)             pairwise similarity, (seq_len, seq_len)

    A     = softmax(scores)                row-wise probability distribution

    output = A V                           weighted blend of values, (seq_len, d_v)

The ``1 / sqrt(d_k)`` factor is not cosmetic. Without it, ``Q K^T`` grows with the
square root of ``d_k``, so the softmax receives large logits and saturates: its
gradient approaches zero and the model stops learning. Dividing by ``sqrt(d_k)``
holds the logits near unit scale regardless of embedding width.

Every step is written out explicitly in the forward pass. No
``torch.nn.MultiheadAttention`` or other high-level layer is called - the point of
this module is that the mathematics is legible and checkable line by line.

Input representation
--------------------
Two modes, both driven by configuration rather than invented here.

``nucleotide`` (the PRD's prescription, Section 6.3)
    A sequence over ``{A, C, G, U}`` mapped to a learnable embedding table, so a
    string of *n* bases becomes an ``(n, embed_dim)`` matrix. This is what FR-2.1
    and FR-2.2 govern, and it is the default.

``feature``
    Each column of the project's processed patient representation becomes one
    token. This satisfies the requirement to accept the existing processed input
    without inventing any variable: the tokens are exactly the encoded columns
    produced by :mod:`src.data.preprocessing`.

What these weights do and do not mean
-------------------------------------
The projections are randomly initialised and untrained. The attention matrix is a
faithful picture of *this arithmetic on this input*, and of nothing else. A high
weight says position *i* assigned a large share of its softmax distribution to
position *j* under these particular random projections - it is not evidence that
one feature influences another, and certainly not evidence of a biological or
clinical relationship. No module in this file draws such a conclusion, and
:func:`summarize_attention` is worded to keep that boundary explicit.

Status: fully implemented. Run ``python -m src.attention.self_attention``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src import config


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SequenceValidationError(ValueError):
    """Base class for rejected attention input."""


class InvalidBaseError(SequenceValidationError):
    """Raised when a sequence contains a character outside {A, C, G, U}.

    FR-2.2 requires an explicit error rather than silent output, so this is
    raised instead of skipping or substituting the offending character.
    """

    def __init__(self, invalid: list[str], sequence: str) -> None:
        self.invalid = list(invalid)
        self.sequence = sequence
        super().__init__(
            f"Invalid nucleotide character(s): {', '.join(self.invalid)}. "
            f"Only {', '.join(config.NUCLEOTIDE_VOCAB)} are allowed. "
            f"Received sequence: {sequence!r}"
        )


class InvalidSequenceLengthError(SequenceValidationError):
    """Raised when a sequence length falls outside the FR-2.1 bounds of 5-50."""

    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__(
            f"Sequence length {length} is outside the supported range "
            f"{config.MIN_SEQUENCE_LENGTH}-{config.MAX_SEQUENCE_LENGTH}."
        )


class InvalidAttentionInputError(ValueError):
    """Raised when a tensor passed to :class:`SelfAttention` has the wrong rank."""


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = config.ATTENTION_SEED) -> int:
    """Seed PyTorch's global RNG and return the seed applied.

    Called before constructing any module so that parameter initialisation - and
    therefore the resulting attention matrix - is reproducible. Uses
    ``config.ATTENTION_SEED`` rather than the training seed so attention
    initialisation can be varied independently of the train/test split.

    Returns
    -------
    int
        The seed that was applied, for recording in the results file.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    return seed


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class NucleotideEmbedding(nn.Module):
    """Learnable embedding table for the four RNA bases.

    Each base maps to one dense vector of ``embed_dim`` values, so a sequence of
    *n* bases becomes an ``(n, embed_dim)`` matrix. This is the representation the
    PRD prescribes in Section 6.3.

    Parameters
    ----------
    vocab:
        Allowed characters, ``config.NUCLEOTIDE_VOCAB`` by default.
    embed_dim:
        Width of each base vector.
    seed:
        Seed applied before ``nn.Embedding`` initialises its table, so the
        vectors are reproducible. Defaults to ``config.ATTENTION_SEED``.
    """

    def __init__(
        self,
        vocab: tuple[str, ...] = config.NUCLEOTIDE_VOCAB,
        embed_dim: int = config.ATTENTION_EMBED_DIM,
        seed: int | None = config.ATTENTION_SEED,
    ) -> None:
        super().__init__()
        self.vocab = tuple(vocab)
        self.embed_dim = int(embed_dim)
        self.index = {base: position for position, base in enumerate(self.vocab)}
        if seed is not None:
            # Seeded inside the constructor so the table is a pure function of
            # (vocab, embed_dim, seed) rather than of global RNG state. Without
            # this, two encoders built in sequence would differ, and the parity
            # check would compare unrelated tables.
            torch.manual_seed(int(seed))
        # A single table: one learnable row per base, shape (vocab, embed_dim).
        self.embedding = nn.Embedding(len(self.vocab), self.embed_dim)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, sequence: str) -> Tensor:
        """Convert a nucleotide string to index tensor ``(seq_len,)``.

        Raises
        ------
        InvalidBaseError
            If any character is outside the vocabulary (FR-2.2).
        """
        invalid = sorted({c for c in sequence if c not in self.index})
        if invalid:
            raise InvalidBaseError(invalid, sequence)
        if not sequence:
            raise SequenceValidationError("Sequence is empty.")
        return torch.tensor(
            [self.index[c] for c in sequence], dtype=torch.long
        )

    def forward(self, sequence: str | Tensor) -> Tensor:
        """Embed a nucleotide string or index tensor to ``(seq_len, embed_dim)``."""
        indices = self.encode(sequence) if isinstance(sequence, str) else sequence
        return self.embedding(indices)

    def extra_repr(self) -> str:
        return f"vocab={''.join(self.vocab)}, embed_dim={self.embed_dim}"


class FeatureTokenEmbedding(nn.Module):
    """Projects the processed feature vector into one token per feature.

    Accepts the project's processed patient representation - the encoded columns
    from :mod:`src.data.preprocessing` - and turns it into a sequence of
    ``n_features`` tokens, each of width ``embed_dim``.

    The projection is a learned ``nn.Linear(1, embed_dim)``, i.e. one scalar
    becomes one token vector. No new patient variables are introduced: the caller
    supplies the columns, and the token labels are passed through unchanged so the
    heatmap can name them.
    """

    def __init__(
        self,
        embed_dim: int = config.ATTENTION_EMBED_DIM,
        seed: int | None = config.ATTENTION_SEED,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        if seed is not None:
            torch.manual_seed(int(seed))
        self.projection = nn.Linear(1, self.embed_dim)

    def forward(self, features: Tensor | list[float] | np.ndarray) -> Tensor:
        """Convert a feature vector to ``(n_features, embed_dim)``.

        Raises
        ------
        InvalidAttentionInputError
            If the input is not one-dimensional.
        """
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim != 1:
            raise InvalidAttentionInputError(
                f"Expected a 1-D feature vector, got shape {tuple(tensor.shape)}."
            )
        return self.projection(tensor.unsqueeze(-1))


# ---------------------------------------------------------------------------
# The attention module
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention, written out step by step.

    Parameters
    ----------
    embed_dim:
        Input width. Must match the encoder's output width.
    d_k:
        Query/key dimensionality. The equation divides by ``sqrt(d_k)``, so this
        sets the softmax temperature.
    d_v:
        Value dimensionality.
        seed:
            Seed applied before the projection matrices are drawn, making the
            constructor a pure function of its arguments. Pass ``None`` to draw
            from the ambient RNG state instead.

    Shape contract
    --------------
    ==============  =====================  ==========================
    Tensor          Shape                  Meaning
    ==============  =====================  ==========================
    ``x``           ``(seq_len, embed)``   input embeddings
    ``q``           ``(seq_len, d_k)``     queries
    ``k``           ``(seq_len, d_k)``     keys
    ``v``           ``(seq_len, d_v)``     values
    ``scores``      ``(seq_len, seq_len)`` scaled pairwise similarity
    ``weights``     ``(seq_len, seq_len)`` row-stochastic attention matrix
    ``output``      ``(seq_len, d_v)``     attended representation
    ==============  =====================  ==========================

    Notes
    -----
    Single-head by design. The PRD lists multi-head attention as an optional bonus
    (Section 6.5), and a single head keeps the equation legible for a viva.

    The projections are stored as explicit ``nn.Parameter`` matrices and applied
    with ``@`` rather than delegated to ``nn.Linear``, so the Q/K/V arithmetic is
    visible in the source.
    """

    def __init__(
        self,
        embed_dim: int = config.ATTENTION_EMBED_DIM,
        d_k: int = config.ATTENTION_D_K,
        d_v: int = config.ATTENTION_D_V,
        bias: bool = False,
        seed: int | None = config.ATTENTION_SEED,
    ) -> None:
        super().__init__()
        embed_dim, d_k, d_v = int(embed_dim), int(d_k), int(d_v)
        if min(embed_dim, d_k, d_v) < 1:
            raise ValueError(
                f"Dimensions must be positive, got embed_dim={embed_dim}, "
                f"d_k={d_k}, d_v={d_v}."
            )

        self.embed_dim = embed_dim
        self.d_k = d_k
        self.d_v = d_v

        # Wq, Wk: (embed_dim, d_k)   Wv: (embed_dim, d_v)
        # init_scale keeps weights small so the initial logits - and therefore the
        # initial attention distribution - stay close to uniform.
        init_scale = config.ATTENTION_INIT_SCALE
        if seed is not None:
            torch.manual_seed(int(seed))
        self.Wq = nn.Parameter(torch.randn(embed_dim, d_k) * init_scale)
        self.Wk = nn.Parameter(torch.randn(embed_dim, d_k) * init_scale)
        self.Wv = nn.Parameter(torch.randn(embed_dim, d_v) * init_scale)

        if bias:
            self.bq = nn.Parameter(torch.zeros(d_k))
            self.bk = nn.Parameter(torch.zeros(d_k))
            self.bv = nn.Parameter(torch.zeros(d_v))
        else:
            self.register_parameter("bq", None)
            self.register_parameter("bk", None)
            self.register_parameter("bv", None)

        # sqrt(d_k): the constant that keeps softmax logits at unit scale.
        self.scaling = math.sqrt(d_k)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the Q, K and V projections of ``x``.

        Q = X Wq + bq, and likewise for K and V.
        """
        if x.ndim != 2:
            raise InvalidAttentionInputError(
                f"Expected input of shape (seq_len, embed_dim), got {tuple(x.shape)}."
            )
        if x.shape[1] != self.embed_dim:
            raise InvalidAttentionInputError(
                f"Expected embed_dim={self.embed_dim}, got {x.shape[1]}."
            )
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        if self.bq is not None:
            q, k, v = q + self.bq, k + self.bk, v + self.bv
        return q, k, v

    def attention_weights(
        self, q: Tensor, k: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(weights, scaled_scores)`` for the given projections.

        scores = (Q K^T) / sqrt(d_k), then a row-wise softmax.

        Parameters
        ----------
        mask:
            Optional boolean tensor broadcastable to ``(seq_len, seq_len)``.
            ``True`` marks positions that may be attended to.
        """
        # (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) / self.scaling
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # softmax over the key dimension, so each row sums to 1 (FR-2.4).
        return torch.softmax(scores, dim=-1), scores

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Run self-attention.

        Parameters
        ----------
        x:
            Input embeddings, ``(seq_len, embed_dim)``.
        mask:
            Optional attention mask, broadcastable to ``(seq_len, seq_len)``.

        Returns
        -------
        tuple
            ``(output, weights)`` where output is ``(seq_len, d_v)`` and weights
            is ``(seq_len, seq_len)``. The weights are returned separately because
            the heatmap is drawn from them.
        """
        q, k, v = self.project(x)
        weights, _ = self.attention_weights(q, k, mask)
        # (seq_len, seq_len) @ (seq_len, d_v) -> (seq_len, d_v)
        output = weights @ v
        return output, weights

    def forward_detailed(self, x: Tensor, mask: Tensor | None = None) -> "AttentionResult":
        """Run self-attention and retain every intermediate tensor.

        Same computation as :meth:`forward`; returns an :class:`AttentionResult`
        that also carries Q, K, V and the pre-softmax scores, which the tests and
        the interpretation utility need.

        Returns
        -------
        AttentionResult
        """
        q, k, v = self.project(x)
        weights, scores = self.attention_weights(q, k, mask)
        output = weights @ v
        return AttentionResult(
            weights=weights,
            output=output,
            q=q,
            k=k,
            v=v,
            scores=scores,
            sequence=_token_labels(x.shape[0]),
            input_mode="embedding",
            embed_dim=self.embed_dim,
            d_k=self.d_k,
            d_v=self.d_v,
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AttentionResult:
    """Everything one attention pass produced.

    Attributes
    ----------
    weights:
        ``(seq_len, seq_len)`` attention matrix. Row *i* is the distribution over
        positions that position *i* attends to; rows sum to 1.0 (FR-2.4).
    output:
        ``(seq_len, d_v)`` attended representation, ``weights @ V``.
    q, k, v:
        The projected Query, Key and Value matrices.
    scores:
        Pre-softmax scaled similarity, ``(Q K^T) / sqrt(d_k)``.
    sequence:
        Per-position token labels, for axis labelling.
    input_mode:
        ``'nucleotide'`` or ``'feature'``.
    """

    weights: Tensor
    output: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    scores: Tensor
    sequence: list[str] = field(default_factory=list)
    input_mode: str = "embedding"
    embed_dim: int = config.ATTENTION_EMBED_DIM
    d_k: int = config.ATTENTION_D_K
    d_v: int = config.ATTENTION_D_V

    @property
    def seq_len(self) -> int:
        return int(self.weights.shape[0])

    def row_sums(self) -> np.ndarray:
        """Row sums of the attention matrix - the FR-2.4 sanity check."""
        return self.weights.detach().cpu().numpy().sum(axis=1)

    def top_pairs(self, n: int = 5) -> list[tuple[int, int, float]]:
        """Strongest attention pairs as ``(from, to, weight)``, descending.

        Reported as an arithmetic fact about the matrix. Not a causal claim.
        """
        matrix = self.weights.detach().cpu().numpy()
        order = np.dstack(np.unravel_index(np.argsort(-matrix, axis=None), matrix.shape))[0]
        return [
            (int(i), int(j), float(matrix[i, j]))
            for i, j in order[: max(1, int(n))]
        ]

    def weight_matrix(self) -> np.ndarray:
        """The attention matrix as a NumPy array, for plotting and saving."""
        return self.weights.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Encoding helpers (FR-2.1, FR-2.2)
# ---------------------------------------------------------------------------


def _token_labels(n: int) -> list[str]:
    """Generic position labels when no real token names are available."""
    return [f"p{i}" for i in range(n)]


#: Colour and typography tokens for the heatmap figure.
#:
#: Declared here rather than imported from ``dashboard.styles``. Architecture
#: rule 5 requires the attention package to stay independent of the interface,
#: and duplicating four strings is a smaller cost than the reverse dependency.
#: ``tests/test_attention.py`` asserts the two definitions agree, so they cannot
#: drift apart unnoticed.
FIGURE_PALETTE: dict[str, str] = {
    "surface": "#FFFFFF",
    "border": "#E4E9F0",
    "text": "#0F172A",
    "heatmap_cmap": "viridis",
}


def _figure_palette() -> dict[str, str]:
    """The heatmap colour tokens, resolved from config so they cannot drift."""
    return FIGURE_PALETTE


def validate_sequence(sequence: str) -> str:
    """Validate a nucleotide string against the FR-2.1 / FR-2.2 rules.

    Checks the vocabulary and then the 5-50 length range.

    Returns
    -------
    str
        The validated sequence, unchanged.

    Raises
    ------
    InvalidBaseError, InvalidSequenceLengthError
    """
    invalid = sorted({c for c in sequence if c not in config.NUCLEOTIDE_VOCAB})
    if invalid:
        raise InvalidBaseError(invalid, sequence)
    if not config.MIN_SEQUENCE_LENGTH <= len(sequence) <= config.MAX_SEQUENCE_LENGTH:
        raise InvalidSequenceLengthError(len(sequence))
    return sequence


def build_embedding_table(
    embed_dim: int = config.ATTENTION_EMBED_DIM, seed: int = config.ATTENTION_SEED
) -> dict[str, np.ndarray]:
    """Return the nucleotide embedding table as plain NumPy arrays.

    A convenience for inspection and for the parity check. The authoritative
    table lives inside :class:`NucleotideEmbedding`.

    Parameters
    ----------
    embed_dim:
        Width of each base vector.
    seed:
        Seed applied before initialisation, so the table is reproducible.

    Returns
    -------
    dict
        ``{'A': array(embed_dim), 'C': ..., 'G': ..., 'U': ...}``.
    """
    set_seed(seed)
    encoder = NucleotideEmbedding(embed_dim=embed_dim)
    return {
        base: encoder.embedding.weight[position].detach().cpu().numpy()
        for position, base in enumerate(config.NUCLEOTIDE_VOCAB)
    }


def encode_sequence(
    sequence: str, table: dict[str, np.ndarray] | None = None
) -> np.ndarray:
    """Turn a nucleotide string into a ``(seq_len, embed_dim)`` matrix (FR-2.1).

    ``encode_sequence('ACGU')`` returns shape ``(4, embed_dim)``.

    The length floor of 5 is deliberately *not* enforced here, because FR-2.1's
    own acceptance example is the four-base string ``'ACGU'``. The documented 5-50
    range is enforced by :func:`validate_sequence`, which callers use for real
    input.

    Parameters
    ----------
    sequence:
        String over {A, C, G, U}.
    table:
        Embedding table from :func:`build_embedding_table`. Built when omitted.

    Returns
    -------
    numpy.ndarray
        Row *i* is the embedding of ``sequence[i]``.

    Raises
    ------
    InvalidBaseError
        If any character is outside {A, C, G, U} (FR-2.2).
    """
    invalid = sorted({c for c in sequence if c not in config.NUCLEOTIDE_VOCAB})
    if invalid:
        raise InvalidBaseError(invalid, sequence)
    if not sequence:
        raise SequenceValidationError("Sequence is empty.")
    if table is None:
        table = build_embedding_table()
    missing = [c for c in sequence if c not in table]
    if missing:
        raise InvalidBaseError(sorted(set(missing)), sequence)
    return np.stack([table[base] for base in sequence]).astype(np.float32)


def encode_sequence_tensor(
    sequence: str, encoder: NucleotideEmbedding | None = None
) -> Tensor:
    """PyTorch counterpart of :func:`encode_sequence`, returning ``(seq_len, embed_dim)``."""
    encoder = encoder or NucleotideEmbedding()
    return encoder(sequence)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_self_attention(
    sequence: str | None = None,
    features: Tensor | list[float] | np.ndarray | None = None,
    labels: list[str] | None = None,
    seed: int = config.ATTENTION_SEED,
    embed_dim: int = config.ATTENTION_EMBED_DIM,
    d_k: int = config.ATTENTION_D_K,
    d_v: int = config.ATTENTION_D_V,
    mask: Tensor | None = None,
) -> AttentionResult:
    """Encode an input and run :class:`SelfAttention`, returning every tensor.

    Exactly one of ``sequence`` or ``features`` must be supplied.

    Parameters
    ----------
    sequence:
        Nucleotide string. Selects ``'nucleotide'`` mode.
    features:
        Processed feature vector. Selects ``'feature'`` mode.
    labels:
        Token labels for the axes. Defaults to the sequence characters, or to the
        encoded column names in feature mode.
    seed:
        Applied before initialisation so the result is reproducible.
    embed_dim, d_k, d_v:
        Attention dimensions.
    mask:
        Optional attention mask.

    Returns
    -------
    AttentionResult

    Raises
    ------
    ValueError
        If neither or both inputs are given.
    """
    if (sequence is None) == (features is None):
        raise ValueError(
            "Supply exactly one of 'sequence' (nucleotide mode) or 'features' "
            "(feature mode)."
        )

    set_seed(seed)

    if sequence is not None:
        validate_sequence(sequence)
        encoder: nn.Module = NucleotideEmbedding(embed_dim=embed_dim, seed=seed)
        x = encoder(sequence)
        token_labels = list(sequence)
        input_mode = "nucleotide"
    else:
        encoder = FeatureTokenEmbedding(embed_dim=embed_dim, seed=seed)
        x = encoder(features)
        token_labels = list(labels) if labels else _token_labels(x.shape[0])
        input_mode = "feature"

    module = SelfAttention(embed_dim=embed_dim, d_k=d_k, d_v=d_v, seed=seed)
    module.eval()  # no dropout, no batchnorm: forward is a pure function
    with torch.no_grad():
        result = module.forward_detailed(x, mask)

    result.sequence = token_labels
    result.input_mode = input_mode
    return result


# ---------------------------------------------------------------------------
# NumPy reference and parity check
# ---------------------------------------------------------------------------


def self_attention_numpy(x: np.ndarray, module: SelfAttention) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the same attention with NumPy, for cross-checking PyTorch.

    Not a second implementation used in production - it exists so the tensor
    results can be verified against an independent path.

    Returns
    -------
    tuple of numpy.ndarray
        ``(weights, output)``.
    """
    wq = module.Wq.detach().cpu().numpy()
    wk = module.Wk.detach().cpu().numpy()
    wv = module.Wv.detach().cpu().numpy()

    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = (q @ k.T) / math.sqrt(module.d_k)
    # Max-subtraction for numerical stability; mathematically a no-op.
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    weights = exponentiated / exponentiated.sum(axis=-1, keepdims=True)
    return weights.astype(np.float32), (weights @ v).astype(np.float32)


def self_attention_torch(x: Tensor, module: SelfAttention) -> Tensor:
    """Return only the attention weights from the PyTorch module.

    Provided so callers that want the matrix alone do not have to unpack a tuple.
    """
    module.eval()
    with torch.no_grad():
        _, weights = module(x)
    return weights


def assert_parity(
    sequence: str | None = None,
    features: Any = None,
    labels: list[str] | None = None,
    seed: int = config.ATTENTION_SEED,
    tol: float = config.ATTENTION_PARITY_TOLERANCE,
) -> float:
    """Compare the PyTorch and NumPy paths and return the largest absolute delta.

    Both paths are seeded identically, so the parameters are the same and only the
    arithmetic differs. A delta above ``tol`` means one path has drifted.

    Returns
    -------
    float
        Maximum absolute difference across the attention matrix.
    """
    if sequence is not None:
        set_seed(seed)
        encoder = NucleotideEmbedding()
        x_tensor = encoder(sequence)
    else:
        set_seed(seed)
        encoder = FeatureTokenEmbedding()
        x_tensor = encoder(features)
    x_numpy = x_tensor.detach().cpu().numpy()

    set_seed(seed)
    module = SelfAttention(embed_dim=encoder.embed_dim)

    module.eval()
    with torch.no_grad():
        _, torch_weights = module(x_tensor)
    numpy_weights, _ = self_attention_numpy(x_numpy, module)

    return float(
        np.max(np.abs(torch_weights.detach().cpu().numpy() - numpy_weights))
    )


# ---------------------------------------------------------------------------
# Interpretation - arithmetic only, no clinical claims
# ---------------------------------------------------------------------------


def summarize_attention(
    result: AttentionResult, top_n: int = 5
) -> dict[str, Any]:
    """Summarise the attention matrix numerically and in plain language.

    Reports where the softmax distribution concentrates. Everything here is an
    arithmetic property of the matrix under these particular random projections:
    the wording deliberately avoids causal or clinical verbs, because attention
    weights from untrained projections establish nothing about influence.

    Parameters
    ----------
    result:
        Output of :func:`run_self_attention`.
    top_n:
        How many strongest pairs to list.

    Returns
    -------
    dict
        ``rows`` (the numeric summary), ``pairs`` (labelled pairs) and
        ``narrative`` (cautious sentences).
    """
    weights = result.weight_matrix()
    row_sums = weights.sum(axis=1)
    pairs = result.top_pairs(top_n)

    # How evenly each row distributes its attention. 1.0 = perfectly uniform.
    entropy = -np.sum(weights * np.log(weights + 1e-12), axis=1)
    max_entropy = np.log(weights.shape[1]) if weights.shape[1] > 1 else 1.0
    uniformity = float(np.mean(entropy / max_entropy)) if max_entropy else 1.0

    labelled_pairs = []
    for from_index, to_index, weight in pairs:
        source = result.sequence[from_index] if result.sequence else f"p{from_index}"
        target = result.sequence[to_index] if result.sequence else f"p{to_index}"
        labelled_pairs.append(
            {"from_index": from_index, "from": source,
             "to_index": to_index, "to": target, "weight": round(weight, 6)}
        )

    strongest = labelled_pairs[0] if labelled_pairs else None
    narrative = _technical_narrative(result, labelled_pairs, uniformity)

    return {
        "input_mode": result.input_mode,
        "seq_len": result.seq_len,
        "rows": {
            "row_sum_min": float(row_sums.min()),
            "row_sum_max": float(row_sums.max()),
            "row_sum_mean": float(row_sums.mean()),
            "normalised_within_tolerance": bool(
                np.allclose(row_sums, 1.0, atol=config.ATTENTION_SUM_TOLERANCE)
            ),
            "mean_uniformity": round(uniformity, 6),
            "max_weight": float(weights.max()),
            "min_weight": float(weights.min()),
            "mean_entropy": float(np.mean(entropy)),
        },
        "pairs": labelled_pairs,
        "strongest_pair": strongest,
        "narrative": narrative,
        "interpretation_limits": (
            "These are arithmetic properties of the attention matrix produced by "
            "untrained random projections. They are not evidence of causal, "
            "biological or clinical relationships, and no medical conclusion may "
            "be drawn from them."
        ),
    }


def _technical_narrative(
    result: AttentionResult,
    pairs: list[dict[str, Any]],
    uniformity: float,
) -> list[str]:
    """Cautious sentences describing the matrix. Returns a list of str."""
    if not pairs:
        return ["No attention pairs were produced."]

    sentences: list[str] = []
    strongest = pairs[0]
    sentences.append(
        f"The largest single weight is {strongest['weight']:.4f}, assigned by "
        f"position {strongest['from_index']} to position {strongest['to_index']}."
    )
    sentences.append(
        f"Across all {result.seq_len} positions the mean normalised entropy is "
        f"{uniformity:.4f}, so the distributions are "
        f"{'close to uniform' if uniformity > 0.9 else 'non-uniform'}; a value near "
        "1.0 is what randomly initialised projections typically produce."
    )
    top_five = sum(pair["weight"] for pair in pairs)
    sentences.append(
        f"The {len(pairs)} largest entries account for {top_five:.4f} of the total "
        f"weight mass of {result.seq_len * result.seq_len} entries."
    )
    sentences.append(
        "These figures describe this matrix under random initialisation only. They "
        "do not establish that any position influences any other, and they carry no "
        "clinical meaning."
    )
    return sentences


def interpret_attention(result: AttentionResult, top_n: int = 5) -> list[str]:
    """Plain-language reading of the heatmap, reusing :func:`summarize_attention`."""
    return summarize_attention(result, top_n)["narrative"]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_attention_heatmap(
    result: AttentionResult,
    save_path: Any = None,
    ax: Any = None,
    annotate: bool | None = None,
) -> tuple[Path, Any]:
    """Draw the attention matrix as a labelled heatmap (FR-2.5).

    Rendered from the real weights returned by the PyTorch module. Both axes carry
    the token labels, so the figure is readable without the surrounding report.

    Annotations are added automatically when the matrix is small enough for them to
    remain legible, and suppressed when it is not.

    Parameters
    ----------
    result:
        Output of :func:`run_self_attention`.
    save_path:
        Destination PNG. Defaults to
        ``config.FIGURES_DIR / config.ATTENTION_HEATMAP_FILENAME``.
    ax:
        Existing axes to draw into. A new figure is created when omitted.
    annotate:
        Force annotation on or off. Chosen from the matrix size when omitted.

    Returns
    -------
    tuple
        ``(saved_path, axes)``. The axes are returned so axis labelling can be
        asserted directly in tests rather than inferred from the image.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Colour and figure tokens live in config, which both this package and the
    # dashboard already depend on. Importing ``dashboard.styles`` for four
    # strings would make the attention module depend on the interface, which
    # architecture rule 5 forbids.
    palette = _figure_palette()

    weights = result.weight_matrix()
    labels = result.sequence or _token_labels(weights.shape[0])

    created = ax is None
    if created:
        fig, ax = plt.subplots(
            figsize=config.ATTENTION_HEATMAP_FIGSIZE, constrained_layout=True
        )
    else:
        fig = ax.get_figure()

    sns.heatmap(
        weights,
        ax=ax,
        cmap=palette["heatmap_cmap"],
        annot=annotate,
        fmt=".2f",
        square=True,
        linewidths=0.3,
        linecolor=palette["border"],
        cbar_kws={"label": "Attention weight", "shrink": 0.85},
    )

    ax.set_title(
        "mRNA Self-Attention Weights",
        fontsize=12,
        fontweight="600",
        pad=14,
        color=palette["text"],
    )
    ax.set_xlabel("Attending TO position", fontsize=9)
    ax.set_ylabel("Attending FROM position", fontsize=9)
    ax.tick_params(
        axis="both", labelsize=config.ATTENTION_TICK_FONT_SIZE
    )
    ax.set_xticklabels(labels, rotation=config.ATTENTION_TICK_ROTATION)
    ax.set_yticklabels(labels, rotation=0)

    destination = (
        Path(save_path) if save_path
        else config.FIGURES_DIR / config.ATTENTION_HEATMAP_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination,
        # FR-5.3's 150 DPI floor, applied to every figure the project saves.
        dpi=config.DASHBOARD_DPI,
        bbox_inches="tight",
        facecolor=palette["surface"],
    )
    if created:
        plt.close(fig)
    return destination, ax


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_attention_artifacts(result: AttentionResult, directory: Any = None) -> dict[str, Path]:
    """Persist the weights and token labels for the dashboard to re-read.

    FR-5.2 requires the heatmap panel to be generated from the saved output rather
    than a hard-coded value, so the matrix and its labels are written here.

    Parameters
    ----------
    result:
        Output of :func:`run_self_attention`.
    directory:
        Destination directory. Defaults to ``config.LOGS_DIR``.

    Returns
    -------
    dict
        Mapping of artefact name to written path.
    """
    target = Path(directory) if directory else config.LOGS_DIR
    target.mkdir(parents=True, exist_ok=True)

    weights_path = target / config.ATTENTION_WEIGHTS_PATH.name
    labels_path = target / config.ATTENTION_TOKEN_LABELS_PATH.name

    np.save(weights_path, result.weight_matrix())
    labels_path.write_text(
        json.dumps(
            {
                "labels": list(result.sequence),
                "input_mode": result.input_mode,
                "seq_len": result.seq_len,
                "embed_dim": int(result.embed_dim),
                "key_dimension": int(result.d_k),
                "value_dimension": int(result.d_v),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # The interface's Panel B reads by the neutral alias, mirroring the ROC
    # figure convention. Written only when absent, so repeat runs do not
    # accumulate duplicates in outputs/.
    alias = config.FIGURES_DIR / config.ATTENTION_HEATMAP_SERVED_NAME
    canonical = config.FIGURES_DIR / config.ATTENTION_HEATMAP_FILENAME
    if canonical.exists() and not alias.exists():
        alias.write_bytes(canonical.read_bytes())

    return {"weights": weights_path, "labels": labels_path}


def save_attention_results(
    result: AttentionResult,
    summary: dict[str, Any],
    heatmap_path: Any = None,
    checkpoint_path: Any = None,
    seed: int = config.ATTENTION_SEED,
    path: Any = None,
) -> Path:
    """Write the machine-readable attention summary to ``outputs/logs/``.

    Every value is computed from the tensors the module produced.

    Parameters
    ----------
    result:
        Output of :func:`run_self_attention`.
    summary:
        Output of :func:`summarize_attention`.
    heatmap_path:
        Where the figure was written, recorded for the interface.
    checkpoint_path:
        Where the state dictionary was written.
    seed:
        The seed applied, for reproducibility.
    path:
        Destination. Defaults to ``config.ATTENTION_RESULTS_PATH``.

    Returns
    -------
    pathlib.Path
    """
    destination = Path(path) if path else config.ATTENTION_RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "input_mode": result.input_mode,
        "input_shape": [int(result.seq_len), int(result.embed_dim)],
        "attention_shape": [int(r) for r in result.weights.shape],
        "output_shape": [int(r) for r in result.output.shape],
        "q_shape": [int(r) for r in result.q.shape],
        "k_shape": [int(r) for r in result.k.shape],
        "v_shape": [int(r) for r in result.v.shape],
        "embedding_dimension": int(result.embed_dim),
        "key_dimension": int(result.d_k),
        "value_dimension": int(result.d_v),
        "scaling_factor": float(math.sqrt(result.d_k)),
        "seed": int(seed),
        "max_attention_pair": summary.get("strongest_pair"),
        "top_pairs": summary.get("pairs"),
        "attention_summary": summary.get("rows"),
        "narrative": summary.get("narrative"),
        "interpretation_limits": summary.get("interpretation_limits"),
        "token_labels": list(result.sequence),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "disclaimer": config.REQUIRED_DISCLAIMER,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def save_checkpoint(module: SelfAttention, path: Any = None) -> Path:
    """Save the module's state dictionary under ``models/``.

    PyTorch's recommended checkpoint format: weights only, with the architecture
    reconstructed from code. Saving a whole pickled module would couple the
    artifact to the class definition and is not needed here - nothing is trained.

    Parameters
    ----------
    module:
        The fitted-instantiation whose parameters to persist.
    path:
        Destination. Defaults to ``config.ATTENTION_MODEL_PATH``.

    Returns
    -------
    pathlib.Path
    """
    destination = Path(path) if path else config.ATTENTION_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": module.state_dict(),
        "embed_dim": module.embed_dim,
        "d_k": module.d_k,
        "d_v": module.d_v,
        "init_seed": config.ATTENTION_SEED,
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: Any = None) -> SelfAttention:
    """Rebuild a :class:`SelfAttention` from a saved checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists, naming the command that produces one.
    """
    destination = Path(path) if path else config.ATTENTION_MODEL_PATH
    if not destination.is_file():
        raise FileNotFoundError(
            f"Attention checkpoint not found at {destination}. "
            "Generate it with: python -m src.attention.self_attention"
        )
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    module = SelfAttention(
        embed_dim=int(payload["embed_dim"]),
        d_k=int(payload["d_k"]),
        d_v=int(payload["d_v"]),
    )
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_shapes(result: AttentionResult) -> list[str]:
    """Shape summary lines for the CLI output."""
    return [
        f"  queries            Q : {tuple(result.q.shape)}",
        f"  keys               K : {tuple(result.k.shape)}",
        f"  values             V : {tuple(result.v.shape)}",
        f"  scaled scores   QK^T : {tuple(result.scores.shape)}",
        f"  attention weights   : {tuple(result.weights.shape)}",
        f"  output      A @ V    : {tuple(result.output.shape)}",
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the attention module end to end and write its artifacts.

    Usage
    -----
    ``python -m src.attention.self_attention``
        Runs the PRD's example sequence, prints the tensor shapes, saves the
        heatmap, the state dictionary and the results file.

    ``--feature``
        Runs in feature mode instead, using the first processed patient record and
        the real encoded feature columns as tokens.

    ``--sequence ACGUACGUACGGCUA``
        Runs a specific nucleotide sequence.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.attention.self_attention",
        description="Demonstrate PyTorch self-attention and persist its artifacts.",
    )
    parser.add_argument("--sequence", default=None, help="Nucleotide sequence to attend over")
    parser.add_argument(
        "--feature", action="store_true", help="Use a processed patient record instead"
    )
    parser.add_argument("--seed", type=int, default=config.ATTENTION_SEED)
    parser.add_argument("--embed-dim", type=int, default=config.ATTENTION_EMBED_DIM)
    parser.add_argument("--d-k", type=int, default=config.ATTENTION_D_K)
    parser.add_argument("--d-v", type=int, default=config.ATTENTION_D_V)
    args = parser.parse_args(argv)

    labels: list[str] | None = None
    features = None
    sequence = args.sequence

    if args.feature:
        sequence = None
        try:
            from src.data import preprocessing

            frame = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
            medians, modes = preprocessing.fit_imputation_strategy(frame)
            cleaned = preprocessing.impute_missing_values(frame, medians, modes)
            x, _ = preprocessing.prepare_features_and_target(cleaned)
            features = x.iloc[0].to_numpy(dtype=float)
            labels = list(x.columns)
        except Exception as exc:  # noqa: BLE001 - demo must not hard-fail
            print(f"  Feature mode unavailable ({exc}); falling back to nucleotide mode.")
            features = None
            labels = None

    if sequence is None and features is None:
        sequence = config.ATTENTION_DEMO_SEQUENCE

    result = run_self_attention(
        sequence=sequence,
        features=features,
        labels=labels,
        seed=args.seed,
        embed_dim=args.embed_dim,
        d_k=args.d_k,
        d_v=args.d_v,
    )

    print()
    print("=" * 72)
    print("mRNA SELF-ATTENTION  -  scaled dot-product mechanism (PyTorch)")
    print("=" * 72)
    print(f"Input mode          : {result.input_mode}")
    print(f"Sequence length     : {result.seq_len}")
    print(f"Embedding dimension : {result.embed_dim}")
    print(f"Key/query dimension : {result.d_k}   (scaling = 1/sqrt({result.d_k}) = {1 / math.sqrt(result.d_k):.4f})")
    print(f"Value dimension     : {result.d_v}")
    print(f"Seed                : {args.seed}")
    print()
    print("TENSOR SHAPES")
    print(f"  input embeddings X : ({result.seq_len}, {result.embed_dim})")
    for line in _format_shapes(result)[1:]:
        print(line)
    print()
    print("ROW SUMS (FR-2.4, each must be 1.0)")
    sums = result.row_sums()
    print(f"  min {sums.min():.8f}   max {sums.max():.8f}   "
          f"allclose(1.0): {bool(np.allclose(sums, 1.0, atol=config.ATTENTION_SUM_TOLERANCE))}")
    print()

    summary = summarize_attention(result)
    print("INTERPRETATION (arithmetic only)")
    for sentence in summary["narrative"]:
        print(f"  - {sentence}")
    print(f"  ! {summary['interpretation_limits']}")
    print()

    heatmap_path, _ = plot_attention_heatmap(result)
    set_seed(args.seed)
    encoder = (
        NucleotideEmbedding(embed_dim=args.embed_dim) if sequence is not None
        else FeatureTokenEmbedding(embed_dim=args.embed_dim)
    )
    checkpoint_module = SelfAttention(
        embed_dim=args.embed_dim, d_k=args.d_k, d_v=args.d_v
    )
    checkpoint_path = save_checkpoint(checkpoint_module)

    artifacts = save_attention_artifacts(result)
    results_path = save_attention_results(
        result, summary, heatmap_path, checkpoint_path, args.seed
    )

    print("ARTIFACTS")
    print(f"  heatmap     : {heatmap_path}")
    print(f"  weights     : {artifacts['weights']}")
    print(f"  labels      : {artifacts['labels']}")
    print(f"  results     : {results_path}")
    print(f"  checkpoint  : {checkpoint_path}")
    print()
    print(f"  {config.REQUIRED_DISCLAIMER}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())