"""Tests for the support policy and diagnostics of :func:`~dyson.util.linalg.matrix_power`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.util import matrix_power, matrix_power_with_info
from dyson.util.linalg import MatrixPowerInfo

if TYPE_CHECKING:
    from dyson.typing import Array


def build(eigvals: list[float] | Array, seed: int = 0) -> Array:
    """Build a real symmetric matrix with a prescribed spectrum."""
    eigvals = np.asarray(eigvals, dtype=float)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(eigvals.size, eigvals.size)))
    return (q * eigvals) @ q.T


class TestSharedSupport:
    """Opting in makes the square root and inverse square root agree on the support.

    Off by default: imposing the inverse's cutoff on the forward power discards contributions of
    order the square root of the cutoff, which is far larger than the cutoff itself.
    """

    def test_default_leaves_the_forward_power_untruncated(self) -> None:
        """By default only a negative power applies the cutoff, as before this change."""
        matrix = build([1.0, 0.5, 1e-14])

        _, info_sqrt = matrix_power_with_info(matrix, 0.5)
        _, info_inv = matrix_power_with_info(matrix, -0.5)

        assert info_sqrt.rank == 3
        assert info_inv.rank == 2

    @pytest.mark.parametrize("small", [1e-14, 1e-12, 1e-11])
    def test_powers_share_a_rank(self, small: float) -> None:
        """Under shared support a negligible direction is dropped by both powers."""
        matrix = build([1.0, 0.5, small])

        _, info_sqrt = matrix_power_with_info(matrix, 0.5, shared_support=True)
        _, info_inv = matrix_power_with_info(matrix, -0.5)

        assert info_sqrt.rank == info_inv.rank == 2
        assert info_sqrt.threshold == info_inv.threshold

    def test_product_is_the_support_projector(self) -> None:
        """``A**0.5 @ A**-0.5`` is the projector onto the shared support.

        With a support chosen per power, the square root keeps a direction the inverse square
        root has discarded, and this product is neither idempotent nor of the right trace.
        """
        matrix = build([1.0, 0.5, 1e-14])

        sqrt, info = matrix_power_with_info(matrix, 0.5, shared_support=True)
        inv_sqrt, _ = matrix_power_with_info(matrix, -0.5)
        product = sqrt @ inv_sqrt

        assert np.allclose(product @ product, product, atol=1e-8)
        assert np.trace(product) == pytest.approx(info.rank, abs=1e-8)

    def test_full_rank_product_is_the_identity(self) -> None:
        """With nothing discarded the product is the identity, as before."""
        matrix = build([1.0, 0.5, 0.25])

        sqrt, _ = matrix_power_with_info(matrix, 0.5, shared_support=True)
        inv_sqrt, _ = matrix_power_with_info(matrix, -0.5)

        assert np.allclose(sqrt @ inv_sqrt, np.eye(3), atol=1e-10)


class TestScaleAwareThreshold:
    """The cutoff must track the scale of the input rather than being absolute."""

    def test_threshold_follows_the_largest_eigenvalue(self) -> None:
        """Scaling the matrix scales the cutoff."""
        _, small = matrix_power_with_info(build([1.0, 0.5, 1e-11]), -0.5)
        _, large = matrix_power_with_info(build(np.array([1.0, 0.5, 1e-11]) * 1e6), -0.5)

        assert large.threshold > small.threshold
        assert large.threshold == pytest.approx(1e-10 + 1e-12 * 1e6)

    def test_absolutely_large_direction_can_still_be_negligible(self) -> None:
        """A direction below the scaled cutoff is dropped even though it is large absolutely.

        At a scale of 1e6 the cutoff is ~1e-6, so 1e-7 goes despite being enormous next to the
        1e-10 absolute cutoff the old policy would have applied.
        """
        _, dropped = matrix_power_with_info(build([1e6, 1e6 / 2, 1e-7]), -0.5)
        _, kept = matrix_power_with_info(build([1e6, 1e6 / 2, 1e-5]), -0.5)

        assert dropped.rank == 2
        assert kept.rank == 3

    def test_absolute_threshold_override(self) -> None:
        """An explicit threshold reproduces the old absolute policy."""
        matrix = build([1.0, 0.5, 1e-8])

        _, default = matrix_power_with_info(matrix, -0.5)
        _, overridden = matrix_power_with_info(matrix, -0.5, threshold=1e-6)

        assert default.rank == 3
        assert overridden.rank == 2
        assert overridden.threshold == 1e-6


class TestNegativeEigenvalues:
    """Rounding-scale negatives are clipped; material ones are refused."""

    @pytest.mark.parametrize("negative", [-1e-14, -1e-12])
    def test_roundoff_negative_is_clipped(self, negative: float) -> None:
        """A negative eigenvalue within tolerance is dropped and reported, not raised on."""
        matrix = build([1.0, 0.5, negative])

        _, info = matrix_power_with_info(matrix, -0.5)

        assert info.rank == 2
        assert info.eigval_min < 0.0

    def test_material_negative_raises_when_strict(self) -> None:
        """A materially negative direction has no real root and is refused under strict.

        The previous implementation masked these away with ``mask &= eigvals > 0`` and returned a
        result whose reported error bore no relation to the discarded direction.
        """
        matrix = build([1.0, 0.5, -0.3])

        with pytest.raises(ValueError, match="not positive semi-definite"):
            matrix_power_with_info(matrix, -0.5, strict=True)
        with pytest.raises(ValueError, match="not positive semi-definite"):
            matrix_power_with_info(matrix, 0.5, strict=True)

    def test_material_negative_warns_by_default(self) -> None:
        """Without strict the direction is dropped, but never silently."""
        matrix = build([1.0, 0.5, -0.3])

        with pytest.warns(UserWarning, match="not positive semi-definite"):
            _, info = matrix_power_with_info(matrix, 0.5)

        assert info.rank == 2
        assert info.material_negative == pytest.approx(-0.3)

    def test_no_material_negative_is_reported_as_zero(self) -> None:
        """A clean input reports no material negativity."""
        _, info = matrix_power_with_info(build([1.0, 0.5, 1e-14]), 0.5)

        assert info.material_negative == 0.0

    def test_tolerance_boundary(self) -> None:
        """The boundary between clipped and refused follows neg_rtol times the scale."""
        matrix = build([1.0, 0.5, -1e-9])

        _, info = matrix_power_with_info(matrix, 0.5, neg_rtol=1e-8, strict=True)
        assert info.rank == 2

        with pytest.raises(ValueError, match="not positive semi-definite"):
            matrix_power_with_info(matrix, 0.5, neg_rtol=1e-12, strict=True)

    def test_integer_powers_accept_indefinite_input(self) -> None:
        """An integer power needs no root, so an indefinite matrix stays legal."""
        matrix = build([1.0, 0.5, -0.3])

        inverse, _ = matrix_power_with_info(matrix, -1)

        assert np.allclose(matrix @ inverse, np.eye(3), atol=1e-10)

    def test_non_hermitian_is_unaffected(self) -> None:
        """The positive semi-definiteness rule applies only to real Hermitian input."""
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(4, 4))

        result, _ = matrix_power_with_info(matrix, 0.5, hermitian=False)

        assert np.allclose(result @ result, matrix, atol=1e-8)


class TestDiagnostics:
    """The reported quantities must be the ones the caller needs."""

    def test_fields_describe_the_decomposition(self) -> None:
        """Every field takes the value the decomposition implies."""
        matrix = build([4.0, 1.0, 1e-14])

        _, info = matrix_power_with_info(matrix, -0.5)

        assert isinstance(info, MatrixPowerInfo)
        assert info.size == 3
        assert info.rank == 2
        assert info.truncated
        assert info.eigval_max == pytest.approx(4.0)
        assert info.eigval_min == pytest.approx(1e-14, abs=1e-13)  # the discarded direction
        assert info.condition == pytest.approx(4.0, rel=1e-6)

    def test_condition_is_over_the_retained_support(self) -> None:
        """The discarded directions must not enter the condition estimate."""
        _, info = matrix_power_with_info(build([1.0, 1e-3, 1e-14]), -0.5)

        assert info.condition == pytest.approx(1e3, rel=1e-6)

    def test_untruncated_reports_no_error(self) -> None:
        """A full-rank decomposition discards nothing and so has no error."""
        _, info = matrix_power_with_info(build([1.0, 0.5, 0.25]), -0.5)

        assert not info.truncated
        assert info.discarded_norm == 0.0
        assert info.error == 0.0


class TestErrorDefinition:
    """The reported error must describe the power, not the input."""

    def test_discarded_input_norm_is_not_the_inverse_root_error(self) -> None:
        """Truncating a 1e-14 direction costs ~1e7 in the inverse square root, not ~1e-14.

        The previous implementation reported the norm of the discarded part of the input for
        every power, so an inverse square root claimed an error of ~1e-14 when the direction it
        refused to invert would have contributed ~1e7.
        """
        matrix = build([1.0, 0.5, 1e-14])

        _, info = matrix_power_with_info(matrix, -0.5)

        assert info.discarded_norm < 1e-13
        assert info.error > 1e4
        # Set by the cutoff, not by the discarded eigenvalue, up to the O(1) norm of the
        # discarded projector.
        assert 1.0 < info.error / info.threshold**-0.5 < 10.0

    def test_positive_power_error_is_small(self) -> None:
        """For a positive power the discarded contribution really is small."""
        matrix = build([1.0, 0.5, 1e-14])

        _, info = matrix_power_with_info(matrix, 0.5, shared_support=True)

        # sqrt(1e-14) = 1e-7, up to the O(1) norm of the discarded projector.
        assert 1e-7 <= info.error < 1e-6

    def test_error_is_finite_for_an_exactly_singular_direction(self) -> None:
        """A zero eigenvalue must not make the reported error infinite."""
        matrix = build([1.0, 0.5, 0.0])

        _, info = matrix_power_with_info(matrix, -0.5)

        assert np.isfinite(info.error)
        assert info.error > 0.0


class TestBackwardsCompatibility:
    """The existing tuple API must keep working."""

    def test_return_error_gives_a_float(self) -> None:
        """The tuple API still yields a float when an error is requested."""
        result, error = matrix_power(build([1.0, 0.5, 1e-14]), -0.5, return_error=True)

        assert result.shape == (3, 3)
        assert isinstance(error, float)

    def test_default_returns_none_for_the_error(self) -> None:
        """The tuple API still yields None when no error is requested."""
        result, error = matrix_power(build([1.0, 0.5, 0.25]), -0.5)

        assert result.shape == (3, 3)
        assert error is None

    def test_well_conditioned_result_is_unchanged(self) -> None:
        """Nothing is discarded for a well-conditioned matrix, so the power is exact."""
        matrix = build([4.0, 1.0, 0.25])

        root, _ = matrix_power(matrix, 0.5)

        assert np.allclose(root @ root, matrix, atol=1e-10)
