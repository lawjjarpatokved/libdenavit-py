"""Regression tests for InteractionDiagram2d.find_x_given_y."""
import numpy as np
import pytest

from libdenavit import InteractionDiagram2d


# A notched outline: the right flank runs out to x = 10, cuts back in toward
# x = 2, then bulges out again, so a horizontal line through the notch meets the
# boundary more than once. Points are listed in polar-angle order because the
# constructor sorts them that way.
NOTCHED_X = [10.0, 9.0, 2.0, 7.0, 5.0, 0.0, -5.0, -8.0, -9.0, 0.0, 6.0]
NOTCHED_Y = [0.0, 1.5, 3.0, 4.5, 6.5, 9.0, 7.0, 3.0, -2.0, -6.0, -3.0]

# A convex diamond, where every horizontal line meets the boundary once on each
# side. Behaviour here must be unchanged by the fix.
DIAMOND_X = [10.0, 0.0, -10.0, 0.0]
DIAMOND_Y = [0.0, 10.0, 0.0, -10.0]


@pytest.fixture
def notched():
    return InteractionDiagram2d(NOTCHED_X, NOTCHED_Y, is_closed=True)


@pytest.fixture
def diamond():
    return InteractionDiagram2d(DIAMOND_X, DIAMOND_Y, is_closed=True)


def crossings_at(diagram, y, peak):
    """Every x where the horizontal line at y meets the boundary."""
    path_x = np.linspace(0, peak, 10)
    path_y = y * np.ones(10)
    found = diagram.find_intersection(path_x, path_y)[0]
    return np.atleast_1d(np.asarray(found, dtype=float))


class TestMultipleCrossings:
    """The defect: a list came back where a number was expected."""

    @pytest.mark.parametrize('y', [3.0, 3.5, 4.5, 5.5, 6.5])
    def test_returns_a_scalar(self, notched, y):
        value = notched.find_x_given_y(y, '+')
        assert isinstance(value, float)
        assert not isinstance(value, (list, tuple, np.ndarray))

    @pytest.mark.parametrize('y', [3.0, 3.5, 4.5, 5.5, 6.5])
    def test_returns_the_outermost_crossing(self, notched, y):
        # Guard: these y values must actually produce the ambiguity, otherwise
        # the test would pass without exercising the fix at all.
        found = crossings_at(notched, y, 1.1 * max(NOTCHED_X))
        assert found.size > 1

        assert notched.find_x_given_y(y, '+') == pytest.approx(found.max())

    def test_known_value_through_the_notch(self, notched):
        # At y = 3 the line meets the boundary at x = 2 (the notch) and x = 8.
        # The capacity is the outer one.
        assert notched.find_x_given_y(3.0, '+') == pytest.approx(8.0)

    def test_result_supports_arithmetic(self, notched):
        # This is the operation that used to raise: run_AASHTO_interaction
        # divides the returned value by a float.
        assert notched.find_x_given_y(3.0, '+') / 2.0 == pytest.approx(4.0)

    def test_negative_direction_returns_the_most_negative(self, notched):
        found = crossings_at(notched, 1.0, 1.1 * min(NOTCHED_X))
        value = notched.find_x_given_y(1.0, '-')
        assert isinstance(value, float)
        assert value == pytest.approx(found.min())


class TestSingleCrossing:
    """The convex case must be untouched by the fix."""

    @pytest.mark.parametrize('y, expected', [(0.0, 10.0), (5.0, 5.0), (-5.0, 5.0)])
    def test_positive_direction(self, diamond, y, expected):
        assert diamond.find_x_given_y(y, str('+')) == pytest.approx(expected)

    @pytest.mark.parametrize('y, expected', [(0.0, -10.0), (5.0, -5.0)])
    def test_negative_direction(self, diamond, y, expected):
        assert diamond.find_x_given_y(y, '-') == pytest.approx(expected)

    def test_returns_scalar(self, diamond):
        assert isinstance(diamond.find_x_given_y(5.0, '+'), float)


class TestArgumentValidation:
    def test_rejects_unknown_direction(self, diamond):
        with pytest.raises(ValueError, match='signX must be positive or negative'):
            diamond.find_x_given_y(5.0, 'sideways')
