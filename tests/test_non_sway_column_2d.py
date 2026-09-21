"""Tests for the effective stiffness argument of the ACI elastic analysis.

run_aci_elastic_second_order_analysis builds a single Elastic section before
loading starts, so its stiffness has to be constant. These tests pin down how
that constant is chosen: the historical hardcoded default, the load-independent
built-in methods, an explicit constant, and a callable.
"""
import numpy as np
import pytest

from conftest import make_rect

from libdenavit import NonSwayColumn2d


ANALYSIS_KWARGS = dict(
    analysis_type='proportional_limit_point',
    P=100.0,
    e=1.0,
    section_id=1,
    disp_incr_factor=1e-5,
    num_steps_vertical=100,
    max_1_4_Mu_limit=False,
    section_factored=True,
)


@pytest.fixture
def column():
    return NonSwayColumn2d(section=make_rect(), length=240.0, et=2.0, eb=2.0,
                           axis='x', apply_minimum_eccentricity=False)


def run(column, **kwargs):
    results = column.run_aci_elastic_second_order_analysis(
        **ANALYSIS_KWARGS, **kwargs)
    return results.applied_axial_load_at_limit_point


def test_default_stiffness_is_the_historical_hardcoded_value(column):
    """Callers that pass nothing must see the pre-existing behaviour."""
    section = column.section
    historical = 0.875 * (0.2 * section.Ec * section.Ig('x')
                          + section.Es * section.Isr('x'))
    assert run(column) == pytest.approx(run(column, EI=historical))


def test_explicit_EI_changes_the_result(column):
    section = column.section
    EcIg = section.Ec * section.Ig('x')
    assert run(column, EI=0.3 * EcIg) != pytest.approx(run(column, EI=0.8 * EcIg))


def test_capacity_increases_with_stiffness(column):
    """A stiffer member buckles later, so capacity must be non-decreasing."""
    section = column.section
    EcIg = section.Ec * section.Ig('x')
    capacities = [run(column, EI=gamma * EcIg)
                  for gamma in (0.3, 0.4, 0.6, 0.8)]
    assert np.all(np.diff(capacities) > 0)


def test_callable_EI_matches_the_constant_it_returns(column):
    section = column.section
    EcIg = section.Ec * section.Ig('x')
    assert run(column, EI=lambda **kwargs: 0.6 * EcIg) == pytest.approx(
        run(column, EI=0.6 * EcIg))


def test_callable_EI_receives_the_section_and_extra_kwargs(column):
    seen = {}

    def stiffness(section=None, axis=None, factor=None, **kwargs):
        seen.update(section=section, axis=axis, factor=factor)
        return factor * section.Ec * section.Ig(axis)

    run(column, EI=stiffness, EI_kwargs={'factor': 0.5})
    assert seen['section'] is column.section
    assert seen['axis'] == 'x'
    assert seen['factor'] == 0.5


@pytest.mark.parametrize('EI_type', ['aci-a', 'aci-b', 'gross'])
def test_load_independent_EI_types_match_EIeff(column, EI_type):
    expected = column.section.EIeff('x', EI_type)
    assert run(column, EI_type=EI_type) == pytest.approx(
        run(column, EI=expected))


@pytest.mark.parametrize(
    'EI_type', ['aci-c', 'jf-a', 'jf-b', 'proposed_1_1', 'proposed_1_2',
                'proposed_2', 'ACI-C', ' aci-c '])
def test_load_dependent_EI_types_are_rejected(column, EI_type):
    """The stiffness is fixed before P is known, so these cannot be honoured."""
    with pytest.raises(ValueError, match='depends on the applied load'):
        run(column, EI_type=EI_type)


def test_non_positive_EI_is_rejected(column):
    with pytest.raises(ValueError):
        run(column, EI=0.0)


def test_non_finite_EI_is_rejected(column):
    with pytest.raises(ValueError):
        run(column, EI=float('nan'))
