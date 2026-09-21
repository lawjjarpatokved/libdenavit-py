"""
Demo for NonSwayColumn2d.

Purpose
-------
This script is meant to be re-run every time non_sway_column_2d.py (or the base
Column2d / RC section code it depends on) is changed, as a quick sanity check
that nothing was broken. It exercises the main public entry points of
NonSwayColumn2d in a realistic order:

  1. Build an RC rectangular section and a NonSwayColumn2d on top of it.
  2. Run a single OpenSees proportional analysis (axial-only, e=0) to get the
     column's maximum axial capacity.
  3. Run a single OpenSees nonproportional analysis at a fixed axial load to
     get the moment capacity at that load.
  4. Generate a full OpenSees-based P-M interaction diagram
     (run_ops_interaction).
  5. Generate an AASHTO LRFD-based P-M interaction diagram
     (run_AASHTO_interaction) for two different EI_type assumptions, for
     comparison against the OpenSees results.
  6. Back-calculate effective EI from both the OpenSees results
     (calculated_EI_ops) and from the AASHTO results treated as "design"
     curves (calculated_EI_design).
  7. Plot everything together and print simple pass/fail sanity checks so the
     script can be used as a regression check without staring at a plot.
"""

import numpy as np
import matplotlib.pyplot as plt

from libdenavit import NonSwayColumn2d, InteractionDiagram2d
from libdenavit.section import RC, Rectangle, ReinfRect

SHOW_PLOTS = True


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f'[{status}] {label}')
    return condition


def main():
    all_ok = True

    # region 1. Build the section and column
    length = 14 * 12  # in.
    fc = 6            # ksi
    fy = 60           # ksi
    axis = 'x'

    conc_cross_section = Rectangle(20, 20)          # 20x20 in. square column
    reinforcement = ReinfRect(16, 16, 3, 3, 0.79)    # #8 bars, 3x3 grid, 16x16 in. spacing
    section = RC(conc_cross_section, reinforcement, fc, fy, 'US', dbt=0.5, s=12)

    col = NonSwayColumn2d(section, length, et=1.0, eb=1.0, axis=axis, dxo=length / 1000)

    section_args = (1, "ElasticPP", "Concrete04_no_confinement", 20, 20)
    section_kwargs = {}
    # endregion

    print('=' * 70)
    print('1. Section and column built')
    print(f'   Ag = {section.Ag:.1f} in^2, Ec = {section.Ec:.0f} ksi')
    print(f'   EIgross = {section.EIgross(axis):.3e} kip-in^2')
    print('=' * 70)

    # region 2. Single proportional (axial-only) analysis
    prop_results = col.run_ops_analysis(
        'proportional_limit_point', e=0,
        section_id=1, section_args=section_args, section_kwargs=section_kwargs,
        disp_incr_factor=1e-6, num_steps_vertical=100)

    P_axial_max = prop_results.applied_axial_load_at_limit_point
    print('2. Axial-only proportional analysis')
    print(f'   Exit message: {prop_results.exit_message}')
    print(f'   P_max = {P_axial_max:.1f} kips')
    all_ok &= check('axial-only analysis produced a finite, positive P_max',
                     np.isfinite(P_axial_max) and P_axial_max > 0)
    # endregion

    # region 3. Single nonproportional analysis at a fixed axial load
    P_fixed = 0.5 * P_axial_max
    nonprop_results = col.run_ops_analysis(
        'nonproportional_limit_point', P=P_fixed,
        section_id=1, section_args=section_args, section_kwargs=section_kwargs,
        disp_incr_factor=1e-5)

    M_at_P_fixed = nonprop_results.maximum_abs_moment_at_limit_point
    print('3. Nonproportional analysis at fixed axial load')
    print(f'   P = {P_fixed:.1f} kips, Exit message: {nonprop_results.exit_message}')
    print(f'   M2_max = {M_at_P_fixed:.1f} kip-in.')
    all_ok &= check('nonproportional analysis produced a finite, positive moment',
                     np.isfinite(M_at_P_fixed) and M_at_P_fixed > 0)
    # endregion

    # region 4. Full OpenSees-based interaction diagram
    num_points = 8
    ops_interaction = col.run_ops_interaction(
        section_id=1, section_args=section_args, section_kwargs=section_kwargs,
        num_points=num_points, prop_disp_incr_factor=1e-6, nonprop_disp_incr_factor=1e-5)

    print('4. OpenSees interaction diagram')
    print(f'   P values: {np.round(ops_interaction["P"], 1)}')
    all_ok &= check('OpenSees interaction diagram has no NaNs',
                     not np.any(np.isnan(ops_interaction['P'])) and
                     not np.any(np.isnan(ops_interaction['M2'])))
    all_ok &= check(f'OpenSees interaction diagram has {num_points} points',
                     len(ops_interaction['P']) == num_points)
    # endregion

    # region 5. AASHTO-based interaction diagrams (two EI assumptions)
    aashto_a = col.run_AASHTO_interaction('aci-a', num_points=num_points, section_factored=False)
    aashto_b = col.run_AASHTO_interaction('aci-b', num_points=num_points, section_factored=False)

    print('5. AASHTO interaction diagrams')
    print(f"   EI_type='aci-a' P values: {np.round(aashto_a['P'], 1)}")
    print(f"   EI_type='aci-b' P values: {np.round(aashto_b['P'], 1)}")
    all_ok &= check("AASHTO 'aci-a' diagram has no NaNs", not np.any(np.isnan(aashto_a['P'])))
    all_ok &= check("AASHTO 'aci-b' diagram has no NaNs", not np.any(np.isnan(aashto_b['P'])))
    # endregion

    # region 6. Back-calculate effective EI
    EI_ops = col.calculated_EI_ops(ops_interaction['P'], ops_interaction['M1'], ops_interaction['M2'])
    EI_design = col.calculated_EI_design(
        ops_interaction['P'], ops_interaction['M1'],
        P_design=aashto_a['P'], M2_design=aashto_a['M2'])

    print('6. Back-calculated effective EI')
    print(f'   EIgross           = {EI_ops["EIgross"]:.3e} kip-in^2')
    print(f'   EI (from OpenSees)= {np.round(EI_ops["Calculated EI"], 0)}')
    print(f'   EI (vs. AASHTO)   = {np.round(EI_design["Calculated EI"], 0)}')
    all_ok &= check('back-calculated EI (ops) never exceeds EIgross',
                     np.all(np.nan_to_num(EI_ops['Calculated EI'], nan=0) <= EI_ops['EIgross'] * 1.0001))
    # endregion

    # region 7. Plot everything and show sanity summary
    if SHOW_PLOTS:
        P_cs, M_cs, _ = section.section_interaction_2d(axis, 100, factored=False, only_compressive=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(M_cs, P_cs, 'k-', label='Cross-section (material) limit')
        ax.plot(ops_interaction['M1'], ops_interaction['P'], 'b--x', label='$M_1$ (OpenSees)')
        ax.plot(ops_interaction['M2'], ops_interaction['P'], 'b-o', label='$M_2$ (OpenSees)')
        ax.plot(aashto_a['M1'], aashto_a['P'], 'r--x', label="$M_1$ (AASHTO, EI='aci-a')")
        ax.plot(aashto_a['M2'], aashto_a['P'], 'r-o', label="$M_2$ (AASHTO, EI='aci-a')")
        ax.set_xlabel('Bending moment (kip-in.)')
        ax.set_ylabel('Axial compression (kips)')
        ax.legend(loc='upper right')
        ax.set_title(f'NonSwayColumn2d interaction diagram (L = {length/12:.1f} ft)')
        fig.tight_layout()
        plt.show()
    # endregion

    print('=' * 70)
    print('ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- see [FAIL] lines above')
    print('=' * 70)

    if not all_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
