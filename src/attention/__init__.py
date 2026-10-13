"""Self-attention subpackage: Transformer mathematics over mRNA sequences.

Boundary rules
--------------
* Independent of the dashboard. This package imports nothing from ``dashboard``
  and writes to ``outputs/`` only through its own documented persistence helpers,
  which the pipeline and the interface then read (architecture rule 5).
* Depends only on ``torch``, ``numpy``, ``matplotlib`` and ``seaborn`` plus
  :mod:`src.config`. It knows nothing about retrieval or LLMs.

Implementation note
-------------------
:class:`~src.attention.self_attention.SelfAttention` writes the equation out step
by step rather than delegating to a high-level attention layer, so every
operation is inspectable in the source. The arithmetic is verified against an
independent NumPy path by
:func:`~src.attention.self_attention.assert_parity`.

Boundary note
-------------
The projections are randomly initialised and untrained. Nothing this package
produces supports a causal, biological or clinical claim, and its interpretation
output says so explicitly.

Requirements owned here
-----------------------
FR-2.1  Encode a nucleotide string (A/C/G/U only, length 5-50) into a
        ``(seq_len, embed_dim)`` matrix.
FR-2.2  Reject any sequence containing a character outside {A, C, G, U} with an
        explicit error.
FR-2.3  Compute the Q, K, V projections and return ``(seq_len, seq_len)``
        attention weights.
FR-2.4  Every row of the weights sums to 1.0 within floating-point tolerance.
FR-2.5  Render the weights as a labelled heatmap saved to ``outputs/figures/``.

The formula implemented
-----------------------
    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

The division by ``sqrt(d_k)`` keeps the softmax logits small enough that the
gradient does not vanish as the embedding dimension grows.
"""

from src.attention import self_attention
from src.attention.self_attention import (
    AttentionResult,
    FeatureTokenEmbedding,
    InvalidAttentionInputError,
    InvalidBaseError,
    InvalidSequenceLengthError,
    NucleotideEmbedding,
    SelfAttention,
    build_embedding_table,
    encode_sequence,
    plot_attention_heatmap,
    run_self_attention,
    save_attention_artifacts,
    save_attention_results,
    save_checkpoint,
    set_seed,
    summarize_attention,
    validate_sequence,
)

# Note on the name above: this package deliberately re-exports the *module*
# ``self_attention`` and not the *function* of the same name. Re-exporting the
# function would shadow the submodule, so ``from src.attention import
# self_attention`` would silently return a function where a module was expected.
# Call the function through the module instead:
#
#     from src.attention import self_attention as attn
#     attn.self_attention(attn.encode_sequence("ACGUACGUACGGCUA"))
#
# ``tests/test_attention.py`` asserts this shadowing does not regress.
# The same reasoning applies to ``pipeline.pipeline``, where the module and the
# ``run_pipeline`` function share a name.

__all__ = [
    "self_attention",
    "SelfAttention",
    "AttentionResult",
    "NucleotideEmbedding",
    "FeatureTokenEmbedding",
    "encode_sequence",
    "validate_sequence",
    "build_embedding_table",
    "run_self_attention",
    "set_seed",
    "summarize_attention",
    "plot_attention_heatmap",
    "save_attention_artifacts",
    "save_attention_results",
    "save_checkpoint",
    "InvalidBaseError",
    "InvalidSequenceLengthError",
    "InvalidAttentionInputError",
]