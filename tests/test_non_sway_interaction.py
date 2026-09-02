"""Tests for the interaction-diagram builders of NonSwayColumn2d.

M1 is the larger of the two applied end moments. Reading it from the top end
alone reports M1 = 0 for a column loaded at the bottom only, which places the
point on the axial axis of the diagram.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_rect

from libdenavit import NonSwayColumn2d
import libdenavit.non_sway_column_2d as nsc


def make_column(et, eb):
    return NonSwayColumn2d(section=make_rect(), length=240.0, et=et, eb=eb,
                           dxo=0.0, axis='x')


def fake_result(top, bot, P=100.0, M2=None):
    """A limit-point result with the given applied end moments."""
    return SimpleNamespace(
        applied_axial_load_at_limit_point=P,
        applied_moment_top_at_limit_point=top,
        applied_moment_bot_at_limit_point=bot,
        maximum_abs_moment_at_limit_point=max(abs(top), abs(bot)) if M2 is None else M2,
        exit_message='Analysis Successful',
        applied_moment_top=[0.0, top],
        applied_moment_bot=[0.0, bot],
        maximum_abs_moment=[0.0, max(abs(top), abs(bot))],
        maximum_abs_disp=[0.0, 1.0],
    )


# (top, bot, expected M1) - the sign convention records the bottom negated
END_MOMENTS = [
    (0.0, -75.0, 75.0),    # bottom end only
    (75.0, 0.0, 75.0),     # top end only
    (75.0, -75.0, 75.0),   # equal ends
    (30.0, -75.0, 75.0),   # bottom governs
    (75.0, -30.0, 75.0),   # top governs
]


@pytest.mark.parametrize('top, bot, expected', END_MOMENTS)
def test_proportional_interaction_M1_uses_the_larger_end(monkeypatch, top, bot, expected):
    monkeypatch.setattr(NonSwayColumn2d, 'run_ops_analysis',
                        lambda self, *a, **k: fake_result(top, bot))
    out = make_column(1.0, 1.0).run_ops_interaction_proportional([1.0])
    assert out['M1'] == pytest.approx([expected])


@pytest.mark.parametrize('top, bot, expected', END_MOMENTS)
def test_nonproportional_interaction_M1_uses_the_larger_end(monkeypatch, top, bot, expected):
    monkeypatch.setattr(NonSwayColumn2d, 'run_ops_analysis',
                        lambda self, *a, **k: fake_result(top, bot))
    # The iP == 0 rung is built from a bare cross section, not the column.
    monkeypatch.setattr(nsc, 'CrossSection2d', lambda *a, **k: SimpleNamespace(
        run_ops_analysis=lambda *a, **k: fake_result(0.0, 0.0, P=0.0, M2=500.0)))

    out = make_column(1.0, 1.0).run_ops_interaction(num_points=3)
    # Rung 0 is the axial-only anchor, rung 2 is the pure-bending anchor.
    assert out['M1'][1] == pytest.approx(expected)


def test_M1_is_a_magnitude(monkeypatch):
    """A negative top moment must not make M1 negative."""
    monkeypatch.setattr(NonSwayColumn2d, 'run_ops_analysis',
                        lambda self, *a, **k: fake_result(-75.0, 0.0))
    out = make_column(1.0, 1.0).run_ops_interaction_proportional([1.0])
    assert out['M1'][0] == pytest.approx(75.0)


def test_axial_only_anchor_has_no_first_order_moment(monkeypatch):
    monkeypatch.setattr(NonSwayColumn2d, 'run_ops_analysis',
                        lambda self, *a, **k: fake_result(0.0, 0.0, M2=10.0))
    monkeypatch.setattr(nsc, 'CrossSection2d', lambda *a, **k: SimpleNamespace(
        run_ops_analysis=lambda *a, **k: fake_result(0.0, 0.0, P=0.0, M2=500.0)))
    out = make_column(1.0, 1.0).run_ops_interaction(num_points=3)
    assert out['M1'][0] == 0.0
