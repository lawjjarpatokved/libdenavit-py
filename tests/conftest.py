"""Shared pytest fixtures and helpers for the libdenavit test suite.

Tests import libdenavit from ``src`` so they exercise the working tree rather
than any copy installed into site-packages.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from libdenavit.section import (  # noqa: E402
    RC, Rectangle, Circle, Obround, ReinfRect, ReinfCirc, ReinfIntersectingLoops,
)


# --- section builders -------------------------------------------------------

def make_rect(confined=False, **kwargs):
    reinf = ReinfRect(14, 24, 3, 3, 0.79)
    reinf.db = 1.0
    extra = dict(dbt=0.5, s=4, fyt=60) if confined else {}
    extra.update(kwargs)
    return RC(Rectangle(30, 20), reinf, 4, 60, 'US', **extra)


def make_circle(confined=False, num_bars=8, **kwargs):
    reinf = ReinfCirc(9, num_bars, 0.79)
    reinf.db = 1.0
    extra = dict(dbt=0.5, s=4, fyt=60) if confined else {}
    extra.update(kwargs)
    return RC(Circle(24), reinf, 4, 60, 'US', **extra)


def make_obround(confined=False, **kwargs):
    reinf = ReinfIntersectingLoops(20, 12, 8, 0.79)
    reinf.db = 1.0
    extra = dict(dbt=0.5, s=4, fyt=60) if confined else {}
    extra.update(kwargs)
    return RC(Obround(24, 12), reinf, 4, 60, 'US', **extra)


@pytest.fixture
def rect():
    return make_rect()


@pytest.fixture
def circle():
    return make_circle()


@pytest.fixture
def obround():
    return make_obround()


# --- OpenSees call recorder -------------------------------------------------

class OpsRecorder:
    """Stand-in for the opensees module that records fiber geometry calls.

    build_ops_fiber_section talks to OpenSees imperatively, so the only way to
    assert on the geometry it generates is to capture the calls.
    """

    def __init__(self):
        self.fibers = []
        self.layers = []
        self.patches = []
        self.materials = []

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop

    def uniaxialMaterial(self, kind, tag, *args):
        self.materials.append({'kind': kind, 'tag': tag, 'args': args})

    def fiber(self, y, z, area, material):
        self.fibers.append((y, z, area, material))

    def layer(self, kind, material, n, area, *coords):
        self.layers.append({'kind': kind, 'material': material, 'n': n,
                            'area': area, 'coords': coords})

    def patch(self, *args):
        self.patches.append(args)

    # -- derived views --
    @property
    def total_fibers(self):
        """Fibers from direct calls plus those a layer expands into.

        Rectangles emit concrete through layer(), other shapes through fiber(),
        so discretisation tests must count both.
        """
        return len(self.fibers) + sum(layer['n'] for layer in self.layers)

    @property
    def defined_material_tags(self):
        return {m['tag'] for m in self.materials}

    @property
    def used_material_tags(self):
        tags = {f[3] for f in self.fibers}
        tags |= {layer['material'] for layer in self.layers}
        tags |= {p[1] for p in self.patches if len(p) > 1}
        return tags

    def materials_of(self, kind):
        return [m for m in self.materials if m['kind'] == kind]

    @property
    def steel_positions(self):
        """Positions of the positive-area (reinforcing bar) fibers."""
        return sorted({round(f[0], 9) for f in self.fibers if f[2] > 0})

    def layer_fibers(self):
        """(position, area) for every fiber implied by the recorded layers.

        OpenSees places the first and last fiber of a straight layer exactly at
        the supplied endpoints, verified against a zeroLengthSection.
        """
        import numpy as np
        out = []
        for layer in self.layers:
            c = layer['coords']
            n = layer['n']
            if n > 1:
                positions = np.linspace(c[0], c[2], n)
            else:
                positions = np.array([(c[0] + c[2]) / 2])
            out.extend((p, layer['area']) for p in positions)
        return out


@pytest.fixture
def record_section():
    """Build a fiber section against a recorder and return the recorder."""
    import importlib

    importlib.import_module('libdenavit.section.RC')
    rc_module = sys.modules['libdenavit.section.RC']
    fiber_module = importlib.import_module('libdenavit.OpenSees.fiber_section')

    def _record(section, axis, nfy=20, nfx=20,
                conc_mat_type='Concrete04_no_confinement', **kwargs):
        recorder = OpsRecorder()
        saved_rc, saved_fiber = rc_module.ops, fiber_module.ops
        rc_module.ops = recorder
        fiber_module.ops = recorder
        try:
            section.build_ops_fiber_section(
                1, 1, 'ElasticPP', conc_mat_type, nfy, nfx, axis=axis, **kwargs)
        finally:
            rc_module.ops = saved_rc
            fiber_module.ops = saved_fiber
        return recorder

    return _record
