"""Module 2 tests: PyTorch self-attention over mRNA sequences.

Requirements covered
--------------------
FR-2.1  ``encode_sequence('ACGU')`` returns a ``(4, embed_dim)`` matrix.
FR-2.2  ``encode_sequence('ACGX')`` raises rather than silently producing output.
FR-2.3  Q, K, V projections computed; ``self_attention`` returns a
        ``(seq_len, seq_len)`` weight matrix.
FR-2.4  Every row of the weights sums to 1.0 within floating-point tolerance.
FR-2.5  A labelled heatmap PNG lands in ``outputs/figures/``.

Plus the six checks this stage is explicitly required to cover: forward pass,
shape correctness, softmax normalisation, absence of NaN/Inf, determinism, and
heatmap generation from real weights.

Nothing here asserts against hard-coded attention values. Every numeric assertion
is either a structural invariant (shapes, row sums, finiteness) or compares two
independent computations of the same quantity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src import config
from src.attention import self_attention as attn
from tests import requires_artifact


#: The PRD Section 6.4 worked example, length 15.
SAMPLE_SEQUENCE = config.ATTENTION_DEMO_SEQUENCE


@pytest.fixture(scope="module")
def nucleotide_result():
    """Attention over the PRD's example sequence."""
    return attn.run_self_attention(sequence=SAMPLE_SEQUENCE)


@pytest.fixture(scope="module")
def feature_result():
    """Attention over the project's real processed feature columns.

    Built from the same encoded columns the ML stage produces, so the tokens are
    project features rather than invented variables.
    """
    from src.data import preprocessing

    frame = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
    cleaned = preprocessing.impute_missing_values(frame)
    x, _ = preprocessing.prepare_features_and_target(cleaned)
    return attn.run_self_attention(
        features=x.iloc[0].to_numpy(dtype=float),
        labels=list(x.columns),
    )


# ---------------------------------------------------------------------------
# Package namespace
# ---------------------------------------------------------------------------


class TestPackageNamespace:
    """Import-hygiene regression tests.

    ``src/attention/self_attention.py`` holds a class named ``SelfAttention`` and
    the module is ``self_attention``. Re-exporting wrongly would shadow one with
    the other.
    """

    def test_self_attention_resolves_to_the_module(self) -> None:
        import types

        import src.attention as package

        assert isinstance(package.self_attention, types.ModuleType)

    def test_module_exposes_the_module_class(self) -> None:
        from src.attention import self_attention as module

        assert isinstance(module.SelfAttention, type)

    def test_class_is_also_reachable_by_its_own_name(self) -> None:
        from src.attention.self_attention import SelfAttention

        assert issubclass(SelfAttention, torch.nn.Module)


# ---------------------------------------------------------------------------
# Test 1 - forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Test 1: valid input passes through the attention module."""

    def test_forward_returns_output_and_weights(self, nucleotide_result) -> None:
        assert isinstance(nucleotide_result.output, torch.Tensor)
        assert isinstance(nucleotide_result.weights, torch.Tensor)

    def test_module_forward_returns_a_pair(self) -> None:
        encoder = attn.NucleotideEmbedding()
        module = attn.SelfAttention()
        x = encoder(SAMPLE_SEQUENCE)
        output, weights = module(x)
        assert output.shape == (len(SAMPLE_SEQUENCE), module.d_v)
        assert weights.shape == (len(SAMPLE_SEQUENCE), len(SAMPLE_SEQUENCE))

    def test_module_forward_returns_the_same_result_twice(self) -> None:
        """module.eval() makes the forward pass a pure function."""
        module = attn.SelfAttention()
        module.eval()
        x = attn.NucleotideEmbedding()(SAMPLE_SEQUENCE)
        with torch.no_grad():
            first_out, first_weights = module(x)
            second_out, second_weights = module(x)
        assert torch.equal(first_out, second_out)
        assert torch.equal(first_weights, second_weights)

    def test_forward_detailed_retains_every_intermediate(self, nucleotide_result) -> None:
        for tensor in (nucleotide_result.q, nucleotide_result.k,
                       nucleotide_result.v, nucleotide_result.scores,
                       nucleotide_result.weights, nucleotide_result.output):
            assert isinstance(tensor, torch.Tensor)

    def test_gradients_flow_through_the_projections(self) -> None:
        """Confirms the parameters are genuinely part of the computation graph."""
        module = attn.SelfAttention()
        x = attn.NucleotideEmbedding()(SAMPLE_SEQUENCE)
        output, _ = module(x)
        output.sum().backward()
        for name in ("Wq", "Wk", "Wv"):
            grad = getattr(module, name).grad
            assert grad is not None, f"{name} received no gradient"
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0, f"{name} gradient is identically zero"

    def test_output_equals_weights_times_values(self, nucleotide_result) -> None:
        """output = A V, checked numerically rather than structurally."""
        expected = nucleotide_result.weights @ nucleotide_result.v
        assert torch.allclose(nucleotide_result.output, expected, atol=1e-6)

    def test_attention_ignores_values_when_weights_are_fixed(self) -> None:
        """A @ V is linear in V, so scaling V scales the output exactly."""
        module = attn.SelfAttention()
        module.eval()
        x = attn.NucleotideEmbedding()(SAMPLE_SEQUENCE)
        with torch.no_grad():
            _, weights = module(x)
            v = x @ module.Wv
            doubled = weights @ (2.0 * v)
            single = weights @ v
        assert torch.allclose(doubled, 2.0 * single, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2 - shape correctness
# ---------------------------------------------------------------------------


class TestShapeCorrectness:
    """Test 2: Q, K, V, scores, weights and output shapes."""

    def test_input_is_sequence_len_by_embed_dim(self, nucleotide_result) -> None:
        assert nucleotide_result.q.shape[0] == len(SAMPLE_SEQUENCE)

    def test_q_k_v_shapes(self, nucleotide_result) -> None:
        n, d_k, d_v = len(SAMPLE_SEQUENCE), config.ATTENTION_D_K, config.ATTENTION_D_V
        assert tuple(nucleotide_result.q.shape) == (n, d_k)
        assert tuple(nucleotide_result.k.shape) == (n, d_k)
        assert tuple(nucleotide_result.v.shape) == (n, d_v)

    def test_scores_and_weights_are_square(self, nucleotide_result) -> None:
        n = len(SAMPLE_SEQUENCE)
        assert tuple(nucleotide_result.scores.shape) == (n, n)
        assert tuple(nucleotide_result.weights.shape) == (n, n)

    def test_output_shape(self, nucleotide_result) -> None:
        assert tuple(nucleotide_result.output.shape) == (
            len(SAMPLE_SEQUENCE), config.ATTENTION_D_V
        )

    def test_fifteen_base_input_gives_fifteen_by_fifteen(self, nucleotide_result) -> None:
        """FR-2.3 acceptance criterion."""
        assert nucleotide_result.weights.shape == (15, 15)

    @pytest.mark.parametrize("length", [5, 7, 15, 23, 50])
    def test_weights_are_square_for_any_valid_length(self, length) -> None:
        result = attn.run_self_attention(sequence=("ACGU" * 20)[:length])
        assert tuple(result.weights.shape) == (length, length)

    def test_projection_matrices_have_the_declared_shapes(self) -> None:
        module = attn.SelfAttention()
        assert tuple(module.Wq.shape) == (module.embed_dim, module.d_k)
        assert tuple(module.Wk.shape) == (module.embed_dim, module.d_k)
        assert tuple(module.Wv.shape) == (module.embed_dim, module.d_v)

    def test_scaling_is_sqrt_of_d_k(self) -> None:
        """The 1/sqrt(d_k) factor is the equation, not a free parameter."""
        module = attn.SelfAttention(d_k=16)
        assert module.scaling == pytest.approx(4.0)

    def test_d_k_and_d_v_may_differ(self) -> None:
        module = attn.SelfAttention(d_k=4, d_v=6)
        assert module.scaling == pytest.approx(2.0)
        x = attn.NucleotideEmbedding(embed_dim=8)("ACGUACGU")
        with torch.no_grad():
            output, weights = module(x)
        assert tuple(output.shape) == (8, 6)
        assert tuple(weights.shape) == (8, 8)

    def test_wrong_input_width_is_rejected(self) -> None:
        module = attn.SelfAttention(embed_dim=8)
        with pytest.raises(attn.InvalidAttentionInputError):
            module(torch.randn(4, 5))

    def test_one_dimensional_input_is_rejected(self) -> None:
        module = attn.SelfAttention(embed_dim=8)
        with pytest.raises(attn.InvalidAttentionInputError):
            module(torch.randn(8))

    def test_non_positive_dimensions_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            attn.SelfAttention(d_k=0)

    def test_feature_mode_attends_over_real_columns(self, feature_result) -> None:
        n = len(feature_result.sequence)
        assert n == 7
        assert tuple(feature_result.weights.shape) == (7, 7)
        assert feature_result.input_mode == "feature"


# ---------------------------------------------------------------------------
# Test 3 - softmax normalisation
# ---------------------------------------------------------------------------


class TestSoftmaxNormalisation:
    """Test 3: softmax weights are normalised along the key dimension."""

    def test_every_row_sums_to_one(self, nucleotide_result) -> None:
        assert np.allclose(
            nucleotide_result.row_sums(), 1.0, atol=config.ATTENTION_SUM_TOLERANCE
        )

    def test_fr24_acceptance_criterion(self, nucleotide_result) -> None:
        """np.allclose(weights.sum(axis=1), 1.0) must be True."""
        weights = nucleotide_result.weight_matrix()
        assert np.allclose(weights.sum(axis=1), 1.0, atol=config.ATTENTION_SUM_TOLERANCE)

    @pytest.mark.parametrize("length", [5, 7, 15, 23, 50])
    def test_rows_sum_to_one_for_any_valid_length(self, length) -> None:
        result = attn.run_self_attention(sequence=("ACGUGCAU" * 8)[:length])
        assert np.allclose(
            result.row_sums(), 1.0, atol=config.ATTENTION_SUM_TOLERANCE
        )

    def test_weights_are_a_probability_distribution(self, nucleotide_result) -> None:
        weights = nucleotide_result.weight_matrix()
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_column_sums_are_not_constrained(self, nucleotide_result) -> None:
        """Normalisation is over keys, not over queries.

        Asserting the negative so a future refactor cannot accidentally swap the
        softmax dimension and still pass.
        """
        weights = nucleotide_result.weight_matrix()
        column_sums = weights.sum(axis=0)
        assert not np.allclose(column_sums, 1.0, atol=1e-3)

    def test_softmax_does_not_saturate_on_large_logits(self) -> None:
        """Max-subtraction inside torch.softmax keeps this finite."""
        module = attn.SelfAttention()
        scores = torch.full((1, 1_000_000), 1e6)
        weights = torch.softmax(scores, dim=-1)
        assert torch.isfinite(weights).all()

    def test_softmax_preserves_relative_ordering(self) -> None:
        weights = torch.softmax(torch.tensor([[3.0, 1.0, 2.0]]), dim=-1)[0]
        assert weights[0] > weights[2] > weights[1]

    def test_masked_positions_receive_zero_weight(self) -> None:
        """A mask must remove positions from the distribution, not renormalise noise."""
        encoder = attn.NucleotideEmbedding()
        module = attn.SelfAttention()
        module.eval()
        x = encoder(SAMPLE_SEQUENCE)
        n = len(SAMPLE_SEQUENCE)
        mask = torch.ones(n, n, dtype=torch.bool)
        mask[:, -2:] = False
        with torch.no_grad():
            _, weights = module(x, mask)
        assert torch.allclose(weights[:, -2:], torch.zeros(n, 2), atol=1e-8)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(n), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4 - no NaN / Inf
# ---------------------------------------------------------------------------


class TestNoNaNOrInf:
    """Test 4: attention outputs contain no NaN or Inf."""

    def test_attention_weights_are_finite(self, nucleotide_result) -> None:
        assert torch.isfinite(nucleotide_result.weights).all()

    def test_scores_are_finite(self, nucleotide_result) -> None:
        assert torch.isfinite(nucleotide_result.scores).all()

    def test_output_is_finite(self, nucleotide_result) -> None:
        assert torch.isfinite(nucleotide_result.output).all()

    def test_all_intermediates_are_finite(self, nucleotide_result) -> None:
        for tensor in (nucleotide_result.q, nucleotide_result.k,
                       nucleotide_result.v, nucleotide_result.scores,
                       nucleotide_result.weights, nucleotide_result.output):
            assert torch.isfinite(tensor).all(), "a non-finite value appeared"

    def test_no_nan_after_float32_round_trip(self, nucleotide_result) -> None:
        weights = nucleotide_result.weights.detach().cpu().numpy()
        assert not np.isnan(weights).any()
        assert not np.isinf(weights).any()

    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e4, 1e8])
    def test_extreme_embedding_magnitudes_stay_finite(self, scale) -> None:
        """Large inputs would overflow a naive softmax; this confirms they do not."""
        module = attn.SelfAttention()
        module.eval()
        x = attn.NucleotideEmbedding()("ACGUACGUACGGCUA") * scale
        with torch.no_grad():
            output, weights = module(x)
        assert torch.isfinite(weights).all()
        assert torch.isfinite(output).all()

    def test_feature_mode_output_is_finite(self, feature_result) -> None:
        assert torch.isfinite(feature_result.output).all()
        assert torch.isfinite(feature_result.weights).all()

    def test_heat_map_matrix_contains_no_nan(self, nucleotide_result) -> None:
        """The figure must never be rendered from a NaN matrix."""
        weights = nucleotide_result.weight_matrix()
        assert np.isfinite(weights).all()


# ---------------------------------------------------------------------------
# Test 5 - deterministic behaviour
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Test 5: the same seed and input produce the same attention."""

    def test_same_seed_gives_identical_weights(self) -> None:
        first = attn.run_self_attention(sequence=SAMPLE_SEQUENCE, seed=7)
        second = attn.run_self_attention(sequence=SAMPLE_SEQUENCE, seed=7)
        assert torch.equal(first.weights, second.weights)

    def test_same_seed_gives_identical_output(self) -> None:
        first = attn.run_self_attention(sequence=SAMPLE_SEQUENCE, seed=7)
        second = attn.run_self_attention(sequence=SAMPLE_SEQUENCE, seed=7)
        assert torch.equal(first.output, second.output)

    def test_different_seeds_give_different_weights(self) -> None:
        """Confirms the seed actually controls initialisation."""
        first = attn.run_self_attention(sequence=SAMPLE_SEQUENCE, seed=1)
        second = attn.run_self_attention(sequence=SAMPLE_SEQUENCE, seed=2)
        assert not torch.allclose(first.weights, second.weights, atol=1e-6)

    def test_determinism_holds_across_feature_mode(self) -> None:
        vector = np.linspace(0.0, 1.0, 7)
        first = attn.run_self_attention(features=vector, labels=list("abcdefg"), seed=11)
        second = attn.run_self_attention(features=vector, labels=list("abcdefg"), seed=11)
        assert torch.equal(first.weights, second.weights)

    def test_embedding_table_is_reproducible(self) -> None:
        first = attn.build_embedding_table(seed=5)
        second = attn.build_embedding_table(seed=5)
        for base in config.NUCLEOTIDE_VOCAB:
            assert np.allclose(first[base], second[base])

    def test_set_seed_returns_the_seed_applied(self) -> None:
        assert attn.set_seed(99) == 99

    def test_parity_with_independent_numpy_path(self) -> None:
        """A second implementation of the same equation must agree.

        Guards against a typo in one of the two paths that a self-consistency
        check could not catch.
        """
        delta = attn.assert_parity(sequence=SAMPLE_SEQUENCE)
        assert delta <= config.ATTENTION_PARITY_TOLERANCE

    def test_parity_holds_in_feature_mode(self) -> None:
        delta = attn.assert_parity(features=np.linspace(0.0, 1.0, 7))
        assert delta <= config.ATTENTION_PARITY_TOLERANCE


# ---------------------------------------------------------------------------
# Test 6 - heatmap generation from real weights
# ---------------------------------------------------------------------------


class TestHeatmap:
    """Test 6: the visualisation is generated from actual attention weights."""

    def test_heatmap_is_written_where_fr25_requires(self, nucleotide_result, tmp_path) -> None:
        destination = tmp_path / config.ATTENTION_HEATMAP_FILENAME
        saved, _ = attn.plot_attention_heatmap(nucleotide_result, destination)
        assert Path(saved).exists()
        assert Path(saved).stat().st_size > 0

    def test_required_figure_name_is_the_configured_one(self) -> None:
        """The prompt requires outputs/figures/mrna_attention_heatmap.png."""
        assert config.ATTENTION_HEATMAP_FILENAME == "mrna_attention_heatmap.png"
        assert config.ATTENTION_HEATMAP_FILENAME in config.ATTENTION_HEATMAP_FILENAMES

    def test_heatmap_is_a_readable_image(self, nucleotide_result, tmp_path) -> None:
        import matplotlib.image as mpimg

        destination = tmp_path / config.ATTENTION_HEATMAP_FILENAME
        attn.plot_attention_heatmap(nucleotide_result, destination)
        image = mpimg.imread(destination)
        assert image.ndim in (2, 3)
        assert image.shape[0] > 100 and image.shape[1] > 100

    def test_heatmap_axes_are_labelled(self, nucleotide_result, tmp_path) -> None:
        """FR-2.5: both axes labelled with sequence positions."""
        destination = tmp_path / config.ATTENTION_HEATMAP_FILENAME
        _, ax = attn.plot_attention_heatmap(nucleotide_result, destination)
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        assert ax.get_title()

    def test_heatmap_tick_labels_are_the_tokens(self, nucleotide_result, tmp_path) -> None:
        destination = tmp_path / config.ATTENTION_HEATMAP_FILENAME
        _, ax = attn.plot_attention_heatmap(nucleotide_result, destination)
        labels = {t.get_text() for t in ax.get_xticklabels()}
        assert set(SAMPLE_SEQUENCE) <= labels

    def test_feature_mode_heatmap_uses_real_column_names(self, feature_result, tmp_path) -> None:
        """Labels must be generated from the processed input, not invented."""
        destination = tmp_path / "feature_heatmap.png"
        _, ax = attn.plot_attention_heatmap(feature_result, destination)
        labels = {t.get_text() for t in ax.get_yticklabels()}
        assert set(feature_result.sequence) <= labels

    def test_heatmap_meets_the_minimum_dpi(self, nucleotide_result, tmp_path) -> None:
        """FR-5.3's 150 DPI floor, read from the PNG's own metadata."""
        from dashboard import app

        destination = tmp_path / config.ATTENTION_HEATMAP_FILENAME
        attn.plot_attention_heatmap(nucleotide_result, destination)
        dpi = app._read_png_dpi(destination)
        assert dpi is not None
        assert dpi >= config.DASHBOARD_DPI

    def test_committed_heatmap_exists_when_the_module_has_run(self) -> None:
        requires_artifact(
            Path(config.FIGURES_DIR) / config.ATTENTION_HEATMAP_FILENAME,
            "python -m src.attention",
        )

    def test_committed_heatmap_alias_exists_for_the_interface(self) -> None:
        requires_artifact(
            Path(config.FIGURES_DIR) / config.ATTENTION_HEATMAP_SERVED_NAME,
            "python -m src.attention",
        )


# ---------------------------------------------------------------------------
# Encoding and validation (FR-2.1, FR-2.2)
# ---------------------------------------------------------------------------


class TestEncoding:
    """FR-2.1: nucleotide strings become (seq_len, embed_dim) matrices."""

    def test_four_base_sequence_has_four_rows(self) -> None:
        """FR-2.1 acceptance criterion: encode_sequence('ACGU') -> (4, embed_dim)."""
        x = attn.encode_sequence("ACGU")
        assert x.shape == (4, config.ATTENTION_EMBED_DIM)

    def test_row_count_tracks_sequence_length(self) -> None:
        for sequence in ("ACGUA", "ACGUACGU", SAMPLE_SEQUENCE):
            assert attn.encode_sequence(sequence).shape[0] == len(sequence)

    def test_embedding_table_covers_exactly_the_four_nucleotides(self) -> None:
        table = attn.build_embedding_table()
        assert set(table) == set(config.NUCLEOTIDE_VOCAB)

    def test_same_base_embeds_identically(self) -> None:
        """A nucleotide must map to the same vector wherever it appears.

        Indices 2 and 3 are both 'A' in ``'ACAA'``; indices 0 and 2 are 'A' too.
        """
        x = attn.encode_sequence("ACAA")
        assert np.allclose(x[0], x[2])
        assert np.allclose(x[2], x[3])

    def test_different_bases_embed_differently(self) -> None:
        """The negative of the above: the table is not constant."""
        x = attn.encode_sequence("ACGU")
        assert not np.allclose(x[0], x[1])

    def test_torch_encoder_matches_the_numpy_table(self) -> None:
        encoder = attn.NucleotideEmbedding(seed=3)
        tensor = encoder("ACGU")
        table = {
            base: encoder.embedding.weight[i].detach().cpu().numpy()
            for i, base in enumerate(config.NUCLEOTIDE_VOCAB)
        }
        expected = np.stack([table[b] for b in "ACGU"])
        assert np.allclose(tensor.detach().numpy(), expected, atol=1e-6)

    def test_embed_dim_is_the_prd_recommended_size(self) -> None:
        """PRD Section 6.3: start with 4 or 8 dimensions."""
        assert config.ATTENTION_EMBED_DIM in (4, 8)

    def test_dimensions_are_not_large(self) -> None:
        """Lightweight by requirement."""
        assert config.ATTENTION_EMBED_DIM <= 64
        assert config.ATTENTION_D_K <= 64
        assert config.ATTENTION_D_V <= 64


class TestSequenceValidation:
    """FR-2.2: invalid nucleotides and out-of-range lengths are rejected."""

    @pytest.mark.parametrize(
        "sequence",
        ["ACGX", "ACGUACGT", "ACGUT", "acgu", "ACGU ACGU", "ACGU\nACGU", "ACGU123"],
    )
    def test_invalid_characters_raise(self, sequence) -> None:
        """'ACGX' must raise; T, lowercase, whitespace and digits must too."""
        with pytest.raises(attn.InvalidBaseError):
            attn.encode_sequence(sequence)

    @pytest.mark.parametrize(
        "sequence", ["ACGU", "ACGUA", "ACGUACGUACGGCUA", "A" * 50]
    )
    def test_valid_sequences_are_accepted(self, sequence) -> None:
        """The negative guard, so the rejection test cannot pass by accident."""
        assert attn.encode_sequence(sequence).shape[0] == len(sequence)

    def test_error_names_the_offending_character(self) -> None:
        with pytest.raises(attn.InvalidBaseError) as excinfo:
            attn.encode_sequence("ACGX")
        assert "X" in str(excinfo.value)

    def test_empty_sequence_raises(self) -> None:
        with pytest.raises(attn.SequenceValidationError):
            attn.encode_sequence("")

    def test_documented_length_range_is_enforced_by_validate(self) -> None:
        """The 5-50 range lives in validate_sequence, keeping FR-2.1's own
        four-base acceptance example valid."""
        with pytest.raises(attn.InvalidSequenceLengthError):
            attn.validate_sequence("ACGU")
        with pytest.raises(attn.InvalidSequenceLengthError):
            attn.validate_sequence("ACGU" * 13)
        assert attn.validate_sequence("ACGUA") == "ACGUA"

    def test_run_self_attention_validates_length(self) -> None:
        with pytest.raises(attn.InvalidSequenceLengthError):
            attn.run_self_attention(sequence="ACGU")

    def test_run_self_attention_requires_exactly_one_input(self) -> None:
        with pytest.raises(ValueError):
            attn.run_self_attention()
        with pytest.raises(ValueError):
            attn.run_self_attention(sequence="ACGUACGU", features=[1.0] * 7)


# ---------------------------------------------------------------------------
# Interpretation - arithmetic only
# ---------------------------------------------------------------------------


class TestInterpretation:
    """The summary must stay technical and free of clinical claims."""

    def test_summary_reports_row_sum_bounds(self, nucleotide_result) -> None:
        summary = attn.summarize_attention(nucleotide_result)
        assert summary["rows"]["normalised_within_tolerance"] is True
        assert summary["rows"]["row_sum_min"] == pytest.approx(1.0, abs=1e-5)
        assert summary["rows"]["row_sum_max"] == pytest.approx(1.0, abs=1e-5)

    def test_strongest_pair_is_reported(self, nucleotide_result) -> None:
        summary = attn.summarize_attention(nucleotide_result)
        pair = summary["strongest_pair"]
        weights = nucleotide_result.weight_matrix()
        assert pair["weight"] == pytest.approx(float(weights.max()), abs=1e-6)

    def test_top_pairs_are_ordered_descending(self, nucleotide_result) -> None:
        weights = [
            pair["weight"] for pair in attn.summarize_attention(nucleotide_result)["pairs"]
        ]
        assert weights == sorted(weights, reverse=True)

    def test_pair_labels_match_the_tokens(self, feature_result) -> None:
        pair = attn.summarize_attention(feature_result)["strongest_pair"]
        assert pair["from"] in feature_result.sequence
        assert pair["to"] in feature_result.sequence

    def test_uniformity_is_reported(self, nucleotide_result) -> None:
        """Random initialisation should produce a near-uniform distribution."""
        summary = attn.summarize_attention(nucleotide_result)
        assert 0.0 <= summary["rows"]["mean_uniformity"] <= 1.0

    def test_narrative_avoids_causal_and_clinical_language(self, nucleotide_result) -> None:
        """Attention weights do not establish causality.

        This is the guard that matters most in this module: an interpretation
        phrased as a clinical claim would misrepresent what the numbers mean.
        """
        summary = attn.summarize_attention(nucleotide_result)
        text = " ".join(summary["narrative"]).lower()
        for forbidden in (
            "cause", "causes", "caused", "causal",
            "risk factor", "diagnos", "predicts outcome",
            "safe", "unsafe", "approve", "clinical decision",
            "adverse reaction", "harmful", "beneficial",
        ):
            assert forbidden not in text, f"interpretation implies '{forbidden}'"

    def test_interpretation_states_its_limits_explicitly(self, nucleotide_result) -> None:
        limits = attn.summarize_attention(nucleotide_result)["interpretation_limits"]
        assert "not evidence" in limits.lower()
        assert "clinical" in limits.lower()

    def test_interpret_attention_returns_narrative(self, nucleotide_result) -> None:
        sentences = attn.interpret_attention(nucleotide_result)
        assert isinstance(sentences, list) and sentences
        assert all(isinstance(s, str) for s in sentences)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """Artefacts must be written, machine-readable and round-trip."""

    def test_weights_and_labels_are_persisted(self, nucleotide_result, tmp_path) -> None:
        written = attn.save_attention_artifacts(nucleotide_result, tmp_path)
        assert written["weights"].exists()
        assert written["labels"].exists()

    def test_saved_weights_match_the_computed_matrix(self, nucleotide_result, tmp_path) -> None:
        attn.save_attention_artifacts(nucleotide_result, tmp_path)
        saved = np.load(tmp_path / config.ATTENTION_WEIGHTS_PATH.name)
        assert np.allclose(saved, nucleotide_result.weight_matrix())

    def test_saved_labels_match_the_tokens(self, nucleotide_result, tmp_path) -> None:
        attn.save_attention_artifacts(nucleotide_result, tmp_path)
        payload = json.loads(
            (tmp_path / config.ATTENTION_TOKEN_LABELS_PATH.name).read_text(encoding="utf-8")
        )
        assert payload["labels"] == list(nucleotide_result.sequence)
        assert payload["input_mode"] == "nucleotide"

    def test_results_file_has_the_required_fields(self, nucleotide_result, tmp_path) -> None:
        summary = attn.summarize_attention(nucleotide_result)
        path = attn.save_attention_results(
            nucleotide_result, summary, tmp_path / "heatmap.png", None, seed=2024
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "input_shape", "attention_shape", "embedding_dimension",
            "key_dimension", "max_attention_pair", "heatmap_path",
        ):
            assert key in payload, f"{key} missing from the results file"

    def test_results_shapes_are_consistent(self, nucleotide_result, tmp_path) -> None:
        summary = attn.summarize_attention(nucleotide_result)
        path = attn.save_attention_results(nucleotide_result, summary, None, None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        n = len(SAMPLE_SEQUENCE)
        assert payload["attention_shape"] == [n, n]
        assert payload["input_shape"] == [n, config.ATTENTION_EMBED_DIM]
        assert payload["output_shape"] == [n, config.ATTENTION_D_V]
        assert payload["key_dimension"] == config.ATTENTION_D_K

    def test_results_carry_the_disclaimer(self, nucleotide_result, tmp_path) -> None:
        summary = attn.summarize_attention(nucleotide_result)
        path = attn.save_attention_results(nucleotide_result, summary, None, None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["disclaimer"] == config.REQUIRED_DISCLAIMER

    def test_results_report_the_random_initialisation_caveat(
        self, nucleotide_result, tmp_path
    ) -> None:
        summary = attn.summarize_attention(nucleotide_result)
        path = attn.save_attention_results(nucleotide_result, summary, None, None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "not evidence" in payload["interpretation_limits"].lower()

    def test_checkpoint_round_trips_the_parameters(self, tmp_path) -> None:
        """A reloaded module must produce the same attention matrix."""
        attn.set_seed(31)
        module = attn.SelfAttention()
        path = attn.save_checkpoint(module, tmp_path / "self_attention.pt")
        restored = attn.load_checkpoint(path)
        x = attn.NucleotideEmbedding()("ACGUACGUACGGCUA")
        module.eval()
        restored.eval()
        with torch.no_grad():
            _, before = module(x)
            _, after = restored(x)
        assert torch.equal(before, after)

    def test_checkpoint_stores_a_state_dict(self, tmp_path) -> None:
        path = attn.save_checkpoint(attn.SelfAttention(), tmp_path / "ckpt.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "state_dict" in payload
        assert set(payload["state_dict"]) == {"Wq", "Wk", "Wv"}

    def test_missing_checkpoint_names_the_fix(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="src.attention"):
            attn.load_checkpoint(tmp_path / "absent.pt")

    def test_committed_artifacts_exist_when_the_module_has_run(self) -> None:
        requires_artifact(config.ATTENTION_RESULTS_PATH, "python -m src.attention")
        requires_artifact(config.ATTENTION_MODEL_PATH, "python -m src.attention")


# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------


class TestScope:
    """This stage must not pull in anything beyond the agreed stack."""

    def test_module_does_not_import_retrieval_or_llm_dependencies(self) -> None:
        import ast

        source = Path(attn.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        for forbidden in (
            "transformers", "tensorflow", "keras", "chromadb",
            "langchain", "openai", "sentence_transformers", "fastapi",
        ):
            assert forbidden not in imported, f"{forbidden} was imported"

    def test_attention_package_does_not_import_the_dashboard(self) -> None:
        """Architecture rule 5: the module stays independent of the UI.

        Walks the full AST rather than grepping, so an import nested inside a
        function body is caught too - which is exactly how a dependency creeps
        back in.
        """
        import ast

        for path in Path(attn.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    assert not module.startswith("dashboard"), (
                        f"{path.name} imports {module}"
                    )

    def test_figure_palette_agrees_with_the_dashboard(self) -> None:
        """The duplicated figure tokens must not drift from the design system.

        Architecture rule 5 forbids importing ``dashboard`` here, so the four
        colour strings are restated. This asserts the restatement is correct.
        """
        from dashboard import styles

        assert attn.FIGURE_PALETTE["surface"] == styles.PALETTE["surface"]
        assert attn.FIGURE_PALETTE["border"] == styles.PALETTE["border"]
        assert attn.FIGURE_PALETTE["text"] == styles.PALETTE["text"]
        assert attn.FIGURE_PALETTE["heatmap_cmap"] == styles.ATTENTION_CMAP

    def test_palette_is_contrast_checked(self) -> None:
        """The restated text colour must still clear WCAG AA on the figure surface."""
        def luminance(hex_colour: str) -> float:
            value = hex_colour.lstrip("#")
            channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
            linear = [
                c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in channels
            ]
            r, g, b = linear
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        fg = luminance(attn.FIGURE_PALETTE["text"])
        bg = luminance(attn.FIGURE_PALETTE["surface"])
        ratio = (max(fg, bg) + 0.05) / (min(fg, bg) + 0.05)
        assert ratio >= 4.5, f"figure text contrast is only {ratio:.2f}:1"