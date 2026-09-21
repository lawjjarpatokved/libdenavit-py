from math import pi, sin, log10
from libdenavit import opensees as ops
from libdenavit import find_limit_point_in_list, interpolate_list, InteractionDiagram2d, CrossSection2d
from libdenavit.OpenSees import AnalysisResults
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import fsolve
import io
import sys
from libdenavit.analysis_helpers import try_analysis_options, ops_get_section_strains, ops_get_maximum_abs_moment, ops_get_maximum_abs_disp, check_analysis_limits, adapt_step_factor
from libdenavit.column_2d import Column2d
import logging

logger = logging.getLogger(__name__)

# EI_type values whose stiffness varies with the applied load, and which
# therefore cannot be used where a single constant stiffness is required.
_LOAD_DEPENDENT_EI_TYPES = frozenset(
    {'aci-c', 'jf-a', 'jf-b', 'proposed_1_1', 'proposed_1_2', 'proposed_2'})


class NonSwayColumn2d(Column2d):

    _INIT_KWARGS = Column2d._INIT_KWARGS | {
        'apply_minimum_eccentricity', 'creep', 'P_sus', 't_sus',
    }
    def __init__(self, section, length, et, eb, **kwargs):
        """
            Represents a non-sway 2D column

            This class defines a non-sway column element with physical parameters such as section properties,
            length, and material properties. It also allows customization of analysis options.

            Parameters:
                section: The section object representing the cross-sectional properties.
                length: The length of the entire column.
                et: The eccentricity at the top of the column.
                eb: The eccentricity at the bottom of the column.
                kwargs: Additional keyword arguments for customization.
                          dxo (float or None, optional): Initial geometric imperfection.
                                                         Default is 0.0. If None, then no imperfection is included.
                          axis (str, optional): Axis. Default is None.
                          ops_n_elem (int, optional): Number of elements for OpenSees analysis. Default is 8.
                          ops_element_type (str, optional): Type of OpenSees element. Default is 'mixedBeamColumn'.
                          ops_geom_transf_type (str, optional): OpenSees geometric transformation type. Default is 'Corotational'.
                          ops_integration_points (int, optional): Number of integration points for OpenSees analysis. Default is 3.
        """

        super().__init__(section, length, **kwargs)
        
        # Store original eccentricities
        self.et_original = et
        self.eb_original = eb
        
        # Apply minimum eccentricity if requested
        self.apply_minimum_eccentricity = kwargs.get('apply_minimum_eccentricity', False)
        
        if self.apply_minimum_eccentricity:
            et, eb = self._apply_aci_minimum_eccentricity(et, eb)
        
        # Physical parameters
        self.et = et
        self.eb = eb
        
        # Specific defaults for non-sway analysis
        self.creep = kwargs.get('creep', False)
        self.P_sus = kwargs.get('P_sus', 0.0)
        self.t_sus = kwargs.get('t_sus', 10000)


    @property
    def _aci_minimum_eccentricity(self):
        """
        ACI 318-19 Section 6.6.4.5.4 minimum eccentricity.

        e_min = 0.6 + 0.03*h  (in.)   /   15 + 0.03*h  (mm)

        h is the section dimension in the direction of bending.
        """
        h = self.section.depth(self.axis)
        if self.section.units == 'us':
            return 0.6 + 0.03 * h  # in.
        return 15 + 0.03 * h  # mm


    @staticmethod
    def _applied_first_order_moment(M_top, M_bot):
        """Largest applied end moment, the M1 of the ACI 6.2.5.3 1.4 check."""
        return max(abs(M_top), abs(M_bot))


    def _apply_aci_minimum_eccentricity(self, et, eb):
        """
        Raise the dominant end eccentricity to the ACI 318-19 minimum.

        Applied once at construction when apply_minimum_eccentricity is set, so
        every analysis sees the corrected loading. This is the only place the
        minimum eccentricity is enforced.
        """
        e_min = self._aci_minimum_eccentricity
        e_max = max(abs(et), abs(eb))

        if e_max < e_min:
            if abs(et) >= abs(eb):
                et = e_min if et >= 0 else -e_min
            else:
                eb = e_min if eb >= 0 else -e_min

        return et, eb


    def _get_default_deformation_limit(self):
        """Override default deformation limit for non-sway columns."""
        return 0.1 * self.length


    def _reference_axial_load(self):
        """
        Reference axial load used to size load-control increments.

        num_steps_vertical is the number of steps taken to reach the reference
        load, so the increment has to be on the scale of the column's axial
        capacity. Using a unit reference makes an axial-only analysis take
        (capacity / num_steps_vertical) times more steps than intended.

        Falls back to a unit reference for sections that define no squash load.
        """
        try:
            p0 = float(self.section.p0)
        except (AttributeError, TypeError, ValueError):
            return 1.0
        if not np.isfinite(p0) or p0 <= 0:
            return 1.0
        return p0


    @property
    def _oriented_ecc(self):
        """
        End eccentricities (et, eb) oriented so the dominant one is non-negative.

        All applied end moments are built from these, so the column always bends
        toward +x -- the direction of the initial imperfection and of the
        displacement-control DOF. Orienting by the dominant (largest magnitude)
        eccentricity rather than by et alone is what keeps a nonzero eb from
        being zeroed out when et == 0.

        Both ends are scaled by the same sign, so magnitudes and relative sign
        are preserved. Callers read the two cases they need straight off the
        returned pair:

            et * eb < 0          -> double curvature
            abs(et) >= abs(eb)   -> the top end dominates
        """
        sign = int(np.sign(max(self.et, self.eb, key=abs)))
        return self.et * sign, self.eb * sign


    def _initialize_results(self):
        """Initialize analysis results object with required attributes."""
        # Get base attributes and add NonSway-specific ones
        results = super()._initialize_results()
        # Add NonSway-specific attributes
        for attr in ['applied_moment_top', 'applied_moment_bot']:
            setattr(results, attr, [])
        return results
    

    def _set_limit_point_values(self, results, ind, x):
        """Override to include moment values."""
        super()._set_limit_point_values(results, ind, x)
        results.applied_moment_top_at_limit_point = interpolate_list(results.applied_moment_top, ind, x)
        results.applied_moment_bot_at_limit_point = interpolate_list(results.applied_moment_bot, ind, x)


    def _run_proportional_analysis(self, config, results):
        """Run proportional analysis for non-sway column."""
        if self.creep:
            return self._run_ops_proportional_with_creep(config, results)
        else:
            return self._run_ops_proportional_no_creep(config, results)


    def _run_nonproportional_analysis(self, config, results):
        """Run nonproportional analysis for non-sway column."""
        return self._run_ops_nonproportional_limit_point(config, results)


    def run_ops_interaction(self, **kwargs):
    
        # Parse keyword arguments
        section_id = kwargs.get('section_id', 1)
        section_args = kwargs.get('section_args', [])
        section_kwargs = kwargs.get('section_kwargs', {})
        num_points = kwargs.get('num_points', 10)
        prop_disp_incr_factor = kwargs.get('prop_disp_incr_factor', 1e-6)
        nonprop_disp_incr_factor = kwargs.get('nonprop_disp_incr_factor', 1e-5)
        section_load_factor = kwargs.get('section_load_factor', 1e-1)
        plot_load_deformation = kwargs.get('plot_load_deformation', False)
        full_results = kwargs.get('full_results', False)

        if plot_load_deformation:
            fig_at_step, ax_at_step = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

        # Run one axial load only analyis to determine maximum axial strength
        results = self.run_ops_analysis('proportional_limit_point', e=0, section_id=section_id,
                                        section_args=section_args, section_kwargs=section_kwargs,
                                        disp_incr_factor=prop_disp_incr_factor, num_steps_vertical=100)
        P = [results.applied_axial_load_at_limit_point]
        M1 = [0]
        M2 = [results.maximum_abs_moment_at_limit_point]
        if full_results:
            M1t_path = [results.applied_moment_top]
            M1b_path = [results.applied_moment_bot]
            M2_path = [results.maximum_abs_moment]
            disp_path = [results.maximum_abs_disp]


        exit_message = [results.exit_message]
        if not np.isfinite(P[0]):
            raise ValueError('Analysis failed at axial only loading')

        # Loop axial linearly spaced axial loads witn non-proportional analyses
        for i in range(1,num_points):
            iP = P[0] * (num_points-1-i) / (num_points-1)
            if iP == 0:
                cross_section = CrossSection2d(self.section, self.axis)
                results = cross_section.run_ops_analysis('nonproportional_limit_point', P=0,
                                                         section_id=section_id, section_args=section_args,
                                                         load_incr_factor=section_load_factor)
                P.append(iP)
                M1.append(results.maximum_abs_moment_at_limit_point)
                M2.append(results.maximum_abs_moment_at_limit_point)
                if full_results:
                    M1t_path.append(results.maximum_abs_moment)
                    M1b_path.append(results.maximum_abs_moment)
                    M2_path.append(results.maximum_abs_moment)
                    disp_path.append([0]*len(results.maximum_abs_moment))

                exit_message.append(results.exit_message)
            else:
                results = self.run_ops_analysis('nonproportional_limit_point', P=iP,
                                                section_id=section_id, section_args=section_args,
                                                disp_incr_factor=nonprop_disp_incr_factor)
                P.append(iP)
                M1.append(self._applied_first_order_moment(
                    results.applied_moment_top_at_limit_point,
                    results.applied_moment_bot_at_limit_point))
                M2.append(results.maximum_abs_moment_at_limit_point)
                if full_results:
                    M1t_path.append(results.applied_moment_top)
                    M1b_path.append(results.applied_moment_bot)
                    M2_path.append(results.maximum_abs_moment)
                    disp_path.append(results.maximum_abs_disp)

                exit_message.append(results.exit_message)

            if plot_load_deformation:
                if iP==0:
                    logger.debug(f'{results.maximum_abs_moment=}')
                else:
                    ax_at_step[0].plot(results.maximum_abs_disp, results.applied_moment_top, '-o', label=f'{iP:,.0f}', markersize=5)
                    ax_at_step[0].legend()
                    ax_at_step[1].plot(results.maximum_abs_disp, results.lowest_eigenvalue, label=f'{iP:,.0f}', markersize=5)
        if plot_load_deformation:
            ax_at_step[0].set_xlabel('Displacement (in)')
            ax_at_step[0].set_ylabel('Applied Moment (kips)')

            ax_at_step[1].set_xlabel('Displacement (in)')
            ax_at_step[1].set_ylabel('Eigenvalue')
            ax_at_step[1].set_ylim(-100,)

            fig_at_step.tight_layout()
            plt.show()
        if full_results:
            return {'P': np.array(P), 'M1': np.array(M1), 'M2': np.array(M2), 'exit_message': exit_message,
                    'M1t_path': M1t_path, 'M1b_path': M1b_path, 'M2_path': M2_path, 'disp_path': disp_path}
        else:
            return {'P': np.array(P), 'M1': np.array(M1), 'M2': np.array(M2), 'exit_message': exit_message}


    def run_ops_interaction_proportional(self, e_list, **kwargs):
        results = [self.run_ops_analysis('proportional_limit_point', e=e, **kwargs) for e
                   in e_list]
        P = [result.applied_axial_load_at_limit_point for result in results]
        M1 = [self._applied_first_order_moment(result.applied_moment_top_at_limit_point,
                                              result.applied_moment_bot_at_limit_point)
              for result in results]
        M2 = [result.maximum_abs_moment_at_limit_point for result in results]

        return {'P': np.array(P), 'M1': np.array(M1), 'M2': np.array(M2)}


    def run_AASHTO_interaction(self, EI_type, **kwargs):
        """
        Perform AASHTO LRFD-based interaction analysis for the column.

            Parameters:
                EI_type (str): The type of effective flexural stiffness of member to use in the analysis.
                num_points (int, optional): The number of points to use in the interaction diagram. Default is 10.
                section_factored (bool, optional): Whether to use factored section properties. Default is True.
                Pc_factor (float, optional): The factor to use in calculating the buckling load. Default is 0.75.
                betadns (float, optional): The ratio of the maximum factored sustained axial load to the total factored axial load
                                        for the same load combination. Default is 0 (short-term loading).

            Note:
              This function uses the notation:
                - M1 to represent the applied first-order moment.
                - M2 to represent the internal second-order moment.
                - This notation differs from the notation used in AASHTO.

            Returns:
            dict: A dictionary containing interaction diagram data:
                - 'P': Array of axial loads
                - 'M1': Array of applied first-order moments
                - 'M2': Array of internal second-order moments
        """
        num_points = kwargs.get('num_points', 10)
        section_factored = kwargs.get('section_factored', True)
        Pc_factor = kwargs.get('Pc_factor', 0.75)
        beta_dns = kwargs.get('beta_dns', 0)

        # Get cross-sectional interaction diagram
        P_id, M_id, _ = self.section.section_interaction_2d(self.axis, 100, factored=section_factored,
                                                            only_compressive=True)
        id2d = InteractionDiagram2d(M_id, P_id, is_closed=False)

        k = 1  # Effective length factor (always one for this non-sway column)

        # Run one axial load only analysis to determine maximum axial strength

        # Compute buckling load based on maximum axial strength (this should be a lower bound)
        
        EIeff = self.section.EIeff(self.axis, EI_type, beta_dns, P=max(P_id), M=0, col=self)
        Pc = pi ** 2 * EIeff / (k * self.length) ** 2
        buckling_load = Pc_factor * Pc

        P_list, M1_list, M2_list, EIeff_list = [], [], [], []

        if buckling_load > max(P_id):
            # Buckling does not happen since the maximum axial strength is less than the lower bound buckling load
            P_list.append(max(P_id))
            M1_list.append(0)
            M2_list.append(0)
            EIeff_list.append(EIeff)

        else:
            # Buckling happens
            if EI_type.lower() in ['aci-a', 'aci-b']:
                P_list.append(buckling_load)
                M1_list.append(0)
                M2_list.append(id2d.find_x_given_y(buckling_load, 'pos'))

            else:
                def error(P):
                    P = P[0]
                    EIeff = self.section.EIeff(self.axis, EI_type, beta_dns, P=P, M=0, col=self)
                    Pc = pi ** 2 * EIeff / (k * self.length) ** 2
                    return P - Pc_factor * Pc

                # Find P such that error = zero
                Pguess = 0.9 * max(P_id)
                solution, _, ier, _ = fsolve(error, Pguess, full_output=True)
                if ier != 1:
                    Pguess = 0.1 * max(P_id)
                    solution, _, ier, _ = fsolve(error, Pguess, full_output=True)
                    if ier != 1:
                        raise Exception("Buckling load calculation did not converge.")
                buckling_load = solution[0]

                # Buckling
                error = []

                max_M_section = max(M_id)
                M2_trials = np.arange(0, max_M_section, max_M_section / 1000)
                for M2 in M2_trials:
                    EIeff = self.section.EIeff(self.axis, EI_type, beta_dns, P=buckling_load, M=M2, col=self)
                    Pc = pi ** 2 * EIeff / (k * self.length) ** 2
                    error.append(buckling_load - Pc_factor * Pc)

                # The trial closest to satisfying P == Pc_factor * Pc, i.e. the
                # smallest residual magnitude -- not the most negative residual.
                M2 = M2_trials[int(np.argmin(np.abs(error)))]

                P_list.append(buckling_load)
                M1_list.append(0)
                M2_list.append(M2)
                EIeff_list.append(self.section.EIeff(self.axis, EI_type, beta_dns, P=buckling_load, M=M2, col=self))

        # Loop axial linearly spaced axial loads with non-proportional analyses
        for i in range(1, num_points):
            iP = 0.999 * P_list[0] * (num_points - i - 1) / (num_points - 1)
            if EI_type.lower() in ['aci-a', 'aci-b']:
                iM2 = id2d.find_x_given_y(iP, 'pos')
                EIeff = self.section.EIeff(self.axis, EI_type, beta_dns)
                k = 1
                Pc = pi ** 2 * EIeff / (k * self.length) ** 2
                delta = max(self.Cm / (1 - (iP) / (Pc_factor * Pc)), 1.0)
                iM1 = iM2 / delta
            else:
                iM2_section = id2d.find_x_given_y(iP, 'pos')
                k = 1  # Effective length factor (always one for this non-sway column)

                trials = []
                for iM1_trial in np.linspace(0.0, iM2_section, 1000):
                    EIeff = self.section.EIeff(self.axis, EI_type, beta_dns, P=iP, M=iM1_trial, col=self)
                    Pc = pi ** 2 * EIeff / (k * self.length) ** 2
                    if Pc_factor * Pc <= iP:
                        # No valid magnifier at this first-order moment. Skip rather
                        # than stop, since a custom EI need not fall monotonically with M.
                        continue

                    delta = max(self.Cm / (1 - (iP) / (Pc_factor * Pc)), 1.0)
                    iM2_trial = delta * iM1_trial

                    # The magnified moment still has to fit inside the section.
                    if iM2_trial <= iM2_section:
                        trials.append((iM1_trial, iM2_trial))

                if trials:
                    iM1, iM2 = max(trials, key=lambda trial: trial[0])
                else:
                    iM1, iM2 = 0.0, 0.0
            P_list.append(iP)
            M1_list.append(iM1)
            M2_list.append(iM2)
            EIeff_list.append(self.section.EIeff(self.axis, EI_type, beta_dns, P=iP, M=iM1, col=self))
        results = {'P':np.array(P_list),'M1':np.array(M1_list),'M2':np.array(M2_list), 'EIeff':np.array(EIeff_list)}
        return results


    @property
    def Cm(self) -> float:
        # Handle axial-only case
        if abs(self.et) < 1e-10 and abs(self.eb) < 1e-10:
            return 1.0  # Single curvature
        
        max_ecc = max(self.et, self.eb, key=abs)
        if abs(max_ecc) < 1e-10:
            return 1.0
        
        return 0.6 + 0.4 * min(self.et, self.eb, key=abs) / max_ecc


    def calculated_EI_ops(self, P_list, M1_list, M2_ops, Pc_factor=1) -> dict:
        """
            Back-calculate the effective flexural stiffness (EI) based on OpenSees results.

            Parameters:
                P_list (array-like): Array of axial loads.
                M1_list (array-like): Array of applied first-order moments.
                M2_list (array-like): Array of internal second-order moments.
                Pc_factor (float, optional): The factor to use in calculating the critical buckling load.
                                            Default is 1.

            Returns:
            dict: A dictionary containing back-calculated EI values for operational load conditions:
                - 'P': Array of axial loads
                - 'M1': Array of applied first-order moments
                - 'EI_ops': Array of back-calculated effective flexural stiffness values
                - 'EIgross': Gross flexural stiffness of the section
        """

        P_list = np.array(P_list)
        M1_list = np.array(M1_list)
        M2_ops = np.array(M2_ops)
        EIgross = self.section.EIgross(self.axis)

        M2_list = []
        EI_list_ops = []

        for P, M1 in zip(P_list, M1_list):
            M2 = np.interp(P, np.flip(P_list), np.flip(M2_ops))
            M2_list.append(M2)

            if M1 == 0 or M1 > M2:
                # M1 == 0 (e.g. the axial-only point): no applied moment to back-calculate
                # EI from, so delta = M2 / M1 is undefined rather than just large.
                EI_list_ops.append(float("nan"))
                continue

            delta = M2 / M1
            if np.isclose(self.Cm / delta, 1.0):
                # e.g. the pure-bending point (P=0, M1==M2==delta==1) combined with Cm==1
                # (uniform moment / single curvature): the moment-magnification denominator
                # (1 - Cm/delta) is exactly zero, so Pc is undefined here.
                EI_list_ops.append(float("nan"))
                continue

            Pc = P / (1-self.Cm/delta) / Pc_factor
            k = 1  # Effective length factor (always one for this non-sway column)
            EI = Pc * (k * self.length / pi) ** 2
            if EI>EIgross:
                EI = EIgross
            EI_list_ops.append(EI)

        return {"P":np.array(P_list), "M1":np.array(M1_list), "M2":np.array(M2_list), "Calculated EI":np.array(EI_list_ops),
                "EIgross":EIgross}


    def calculated_EI_design(self, P_list, M1_list, P_design, M2_design, section_factored=False, Pc_factor=1) -> dict:
        """
            Back-calculate the effective flexural stiffness (EI) based on OpenSees and AASHTO values.

            Parameters:
                P_list (array-like): Array of axial loads.
                M1_list (array-like): Array of applied first-order moments.
                Pc_factor (float, optional): The factor to use in calculating the critical buckling load.
                                            Default is 1.

            Returns:
            dict: A dictionary containing back-calculated EI values for operational load conditions:
                - 'P': Array of axial loads
                - 'M1': Array of applied first-order moments
                - 'EI_AASHTO': Array of back-calculated effective flexural stiffness values
                - 'EIgross': Gross flexural stiffness of the section
        """

        P_list = np.array(P_list)
        M1_list = np.array(M1_list)
        P_design = np.array(P_design)
        M2_design = np.array(M2_design)

        EIgross = self.section.EIgross(self.axis)
        M2_list = []
        EI_list_AASHTO = []

        for P, M1 in zip(P_list, M1_list):
            M2 = np.interp(P, np.flip(P_design), np.flip(M2_design))
            M2_list.append(M2)

            if P < min(P_design) or P > max(P_design) or M1 == 0 or M1 > M2:
                # M1 == 0 (e.g. the axial-only point): no applied moment to back-calculate
                # EI from, so delta = M2 / M1 is undefined rather than just large.
                EI_list_AASHTO.append(float("nan"))
                continue

            delta = M2 / M1
            if np.isclose(self.Cm / delta, 1.0):
                # e.g. the pure-bending point (P=0, M1==M2==delta==1) combined with Cm==1
                # (uniform moment / single curvature): the moment-magnification denominator
                # (1 - Cm/delta) is exactly zero, so Pc is undefined here.
                EI_list_AASHTO.append(float("nan"))
                continue

            Pc = P / (1 - self.Cm / delta) / Pc_factor
            k = 1  # Effective length factor (always one for this non-sway column)
            EI = Pc * (k * self.length / pi) ** 2
            if EI>EIgross:
                EI = EIgross
            EI_list_AASHTO.append(EI)
            
        return {"P": np.array(P_list), "M1": np.array(M1_list), "M2":np.array(M2_list), "Calculated EI": np.array(EI_list_AASHTO),
                "EIgross": EIgross}
        
    
    @property
    def ops_mid_node(self):
        if self.ops_n_elem % 2 == 0:
            return self.ops_n_elem // 2
        raise ValueError(f'Number of elements should be even {self.ops_n_elem = }')


    def build_ops_model(self, section_id, section_args, section_kwargs, **kwargs):
        """
           Build the OpenSees finite element model for the non-sway 2D column.

           This method constructs the finite element model in OpenSees for the non-sway 2D column element.

           Parameters:
               section_id: An integer id for the section
               section_args: Positional arguments for building the section using OpenSees via section.build_ops_fiber_section().
                             (For RC sections the args are: section_id, start_material_id, steel_mat_type, conc_mat_type, nfy, nfx)
               section_kwargs: Keword arguments for building the section using OpenSees via section.build_ops_fiber_section().
                               (For RC sections, no kwargs are necessary).
               kwargs: Additional keyword arguments.
                         start_node_fixity (tuple, optional): Fixity conditions at the start node. Default is (1, 1, 0).
                         end_node_fixity (tuple, optional): Fixity conditions at the end node. Default is (1, 0, 0).

           Returns:
               None
       """

        # region Extract kwargs
        creep_props_dict = kwargs.get('creep_props_dict', dict())
        shrinkage_props_dict = kwargs.get('shrinkage_props_dict', dict())
        start_node_fixity = kwargs.get('start_node_fixity', (1, 1, 0))
        end_node_fixity = kwargs.get('end_node_fixity', (1, 0, 0))
        # endregion

        # region Build OpenSees model
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        # endregion

        # region Define Nodes and Fixities and Geometric Transformation
        for index in range(self.ops_n_elem + 1):
            if isinstance(self.dxo, (int, float)):
                x = sin(index / self.ops_n_elem * pi) * self.dxo
            elif self.dxo == None:
                x = 0.
            else:
                raise ValueError(f'Unknown value of dxo ({self.dxo})')
            y = index / self.ops_n_elem * self.length
            ops.node(index, x, y)
            ops.mass(index, 1, 1, 1)

        ops.fix(0, *start_node_fixity)
        ops.fix(self.ops_n_elem, *end_node_fixity)

        ops.geomTransf(self.ops_geom_transf_type, 100)
        # endregion and

        # region Define Fiber Section
        if type(self.section).__name__ == "RC":
            self.section.build_ops_fiber_section(section_id, *section_args, **section_kwargs, axis=self.axis,
                                                 creep=self.creep, creep_props_dict=creep_props_dict,
                                                 shrinkage_props_dict=shrinkage_props_dict)
        elif type(self.section).__name__ == "CCFT":
            self.section.build_ops_fiber_section(section_id, *section_args, **section_kwargs, creep=self.creep, axis=self.axis)
        elif type(self.section).__name__ == "I_shape":
            self.section.build_ops_fiber_section(section_id, *section_args, **section_kwargs, axis=self.axis)
        else:
            raise ValueError(f'Unknown cross section type {type(self.section).__name__}')
        # endregion

        ops.beamIntegration("Lobatto", 1, 1, self.ops_integration_points)

        for index in range(self.ops_n_elem):
            if self.ops_element_type == 'elasticBeamColumn':
                raise NotImplementedError('Elastic beam column element is not implemented for this class.')
            else:
                ops.element(self.ops_element_type, index, index, index + 1, 100, 1)


    def _run_ops_proportional_no_creep(self, config, results):
        """Run proportional limit point analysis without creep."""
        # time = LFV
        ops.timeSeries('Linear', 100)
        ops.pattern('Plain', 200, 100)

        
        et, eb = self._oriented_ecc

        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.test('NormUnbalance', 1e-3, 10)
        ops.algorithm('Newton')
        

        if config['e'] == 0.0:
            # axial-only: vertical load only, no moments
            ops.load(self.ops_n_elem, 0, -1, 0)  # reference vertical load
            ops.load(0, 0, 0, 0.0)                 # no moment
            if config['axial_control'] == 'displacement':
                ops.integrator('DisplacementControl', self.ops_n_elem, 2,
                               -self.length * config['disp_incr_factor'])
            else:
                dF = self._reference_axial_load() / max(1, config['num_steps_vertical'])
                ops.integrator('LoadControl', dF)
        else:
            ops.load(self.ops_n_elem, 0, -1, et * config['e'])
            ops.load(0, 0, 0, -eb * config['e'])
        
            if et * eb < 0:
                if abs(et) >= abs(eb):
                    dof = 3 * self.ops_n_elem // 4
                else:
                    dof = 1 * self.ops_n_elem // 4
                dU = self.length * config['disp_incr_factor'] / 2
                ops.integrator('DisplacementControl', dof, 1, dU)
            else:
                dU = self.length * config['disp_incr_factor']
                ops.integrator('DisplacementControl', self.ops_mid_node, 1, dU)
        
        ops.analysis('Static', '-noWarnings')

        # Define recorder
        def record():
            time = ops.getTime()
            section_strains = ops_get_section_strains(self)

            results.applied_axial_load.append(time)
            results.applied_moment_top.append(et * config['e'] * time)
            results.applied_moment_bot.append(-eb * config['e'] * time)
            results.maximum_abs_moment.append(ops_get_maximum_abs_moment(self))
            results.maximum_abs_disp.append(ops_get_maximum_abs_disp(self))
            results.lowest_eigenvalue.append(ops.eigen('-fullGenLapack', 1)[0])
            if type(self.section).__name__ == "RC":
                results.maximum_concrete_compression_strain.append(section_strains[0])
                results.maximum_steel_strain.append(section_strains[1])
            elif type(self.section).__name__ == "I_shape":
                results.maximum_compression_strain.append(section_strains[0])
                results.maximum_tensile_strain.append(section_strains[1])

            if self.axis == 'x':
                results.curvature.append(section_strains[2])
            elif self.axis == 'y':
                results.curvature.append(section_strains[3])
            else:
                raise ValueError(f'The value of axis ({self.axis}) is not supported.')

        def update_dU(disp_incr_factor, div_factor=1):
            base_incr_factor = config['disp_incr_factor']
            step_scale = disp_incr_factor / base_incr_factor if base_incr_factor else 1.0

            if config['e'] == 0.0:
                if config['axial_control'] == 'displacement':
                    dU = -self.length * disp_incr_factor / div_factor
                    ops.integrator('DisplacementControl', self.ops_n_elem, 2, dU)
                    return
                # axial-only: shrink the load step, stay in LoadControl
                dF = (self._reference_axial_load()
                      / max(1, config['num_steps_vertical'])
                      * step_scale / div_factor)
                ops.integrator('LoadControl', dF)
                return
            et, eb = self._oriented_ecc
            if et * eb < 0:
                if abs(et) >= abs(eb):
                    dof = 3 * self.ops_n_elem // 4
                else:
                    dof = 1 * self.ops_n_elem // 4
                dU = self.length * disp_incr_factor / 2 / div_factor
                ops.integrator('DisplacementControl', dof, 1, dU)
            else:
                dU = self.length * disp_incr_factor / div_factor
                ops.integrator('DisplacementControl', self.ops_mid_node, 1, dU)

        def reset_analysis_options(disp_incr_factor):
            update_dU(disp_incr_factor)
            ops.algorithm('Newton')
            ops.test('NormUnbalance', 1e-3, 10)

        record()


        maximum_applied_axial_load = 0.
        base_incr_factor = config['disp_incr_factor']
        disp_incr_factor = base_incr_factor
        
        
        while True:
            ok = ops.analyze(1)
            recovered_div = None

            if ok != 0 and config['try_smaller_steps']:
                for div_factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
                    update_dU(disp_incr_factor, div_factor)
                    ok = ops.analyze(1)
                    if ok != 0:
                        ok = try_analysis_options()
                    if ok == 0:
                        recovered_div = div_factor
                        break

            if ok != 0 and not config['try_smaller_steps']:
                ok = try_analysis_options()

            if ok == 0 and config['try_smaller_steps']:
                disp_incr_factor = adapt_step_factor(disp_incr_factor, base_incr_factor,
                                                     recovered_div)
                reset_analysis_options(disp_incr_factor)
            elif ok != 0:
                results.exit_message = 'Analysis Failed'
                warnings.warn('Analysis Failed')
                break

            record()

            # Check for drop in applied load
            if config['percent_load_drop_limit'] is not None:
                current_applied_axial_load = results.applied_axial_load[-1]
                maximum_applied_axial_load = max(maximum_applied_axial_load, current_applied_axial_load)
                if current_applied_axial_load < (1 - config['percent_load_drop_limit']) * maximum_applied_axial_load:
                    results.exit_message = 'Load Drop Limit Reached'
                    break

            # Check for limits
            exit_message = check_analysis_limits(
                    results,
                    eigenvalue_limit=config['eigenvalue_limit'],
                    deformation_limit=config['deformation_limit'],
                    concrete_strain_limit=config['concrete_strain_limit'],
                    steel_strain_limit=config['steel_strain_limit'],
                    section_type=type(self.section).__name__,
                )
            if exit_message:
                results.exit_message = exit_message
                break

        return results


    def _run_ops_proportional_with_creep(self, config, results):
        """Run proportional limit point analysis with creep."""
        # region Determine the sign of the eccentricity
        sgn_et = int(np.sign(self.et))
        sgn_eb = int(np.sign(self.eb))
        et, eb = self._oriented_ecc
        # endregion

        # region Define recorder
        def record(lam=0):
            section_strains = ops_get_section_strains(self)

            # Backup the original stderr
            original_stderr = sys.stderr
            try:
                # Redirect stderr to nowhere
                sys.stderr = io.StringIO()
                time = ops.getLoadFactor(200) + ops.getLoadFactor(2000)
            except Exception as e:
                logger.warning(f'Failed to get combined load factor: {e}')
                try:
                    time = ops.getLoadFactor(200)
                except Exception as e:
                    logger.warning(f'Failed to get load factor for pattern 200: {e}')
                    time = 0
            finally:
                # Restore stderr
                sys.stderr = original_stderr

            results.applied_axial_load.append(time + lam)
            results.applied_moment_top.append(et * config['e'] * (time + lam))
            results.applied_moment_bot.append(-eb * config['e'] * (time + lam))
            results.maximum_abs_moment.append(ops_get_maximum_abs_moment(self))
            results.maximum_abs_disp.append(ops_get_maximum_abs_disp(self))
            results.lowest_eigenvalue.append(ops.eigen('-fullGenLapack', 1)[0])
            
            if type(self.section).__name__ == "RC":
                results.maximum_concrete_compression_strain.append(section_strains[0])
                results.maximum_steel_strain.append(section_strains[1])
            elif type(self.section).__name__ == "I_shape":
                results.maximum_compression_strain.append(section_strains[0])
                results.maximum_tensile_strain.append(section_strains[1])

            if self.axis == 'x':
                results.curvature.append(section_strains[2])
            elif self.axis == 'y':
                results.curvature.append(section_strains[3])
            else:
                raise ValueError(f'The value of axis ({self.axis}) is not supported.')
        # endregion

        # region Run the sustained load phase
        load_step_for_sustained = 1000

        ops.setCreep(0)
        ops.setTime(0)
        ops.wipeAnalysis()
        ops.system('Umfpack')
        ops.test('EnergyIncr', 1e-4, 100, 0, 2)
        ops.algorithm('NewtonLineSearch', '-maxIter', 50, '-maxEta ', 1)
        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.integrator('LoadControl', self.P_sus/load_step_for_sustained)
        ops.analysis('Static','-noWarnings')
        
        t = self.section.Tcr
        tfinish = self.t_sus + self.section.Tcr

        ops.timeSeries('Linear', 100)
        ops.pattern('Plain', 200, 100, '-factor', 1)
        ops.load(self.ops_n_elem, 0, -1, et * config['e'])
        ops.load(0, 0, 0, -eb * config['e'])
        
        while t < tfinish:
            # Apply sustained load at Tcr
            if t == self.section.Tcr:
                for i in np.arange(1, load_step_for_sustained+1, 1):
                                        
                    ok = ops.analyze(1)
                    record()

                    
                    if ok < 0:
                        logger.warning(f'Analysis failed before full sustained load is reached, load: {i/load_step_for_sustained*self.P_sus}')
                        results.exit_message = 'Analysis failed before full sustained load is reached'
                        return results

                    if config['deformation_limit'] is not None:
                        if results.maximum_abs_disp[-1] > config['deformation_limit']:
                            logger.warning(f'Analysis failed before full sustained load is reached, load: {i/load_step_for_sustained*self.P_sus}')
                            results.exit_message = 'Analysis failed before full sustained load is reached'
                            return results

                
                
                ops.loadConst('-time', self.P_sus)
                
                ops.integrator('LoadControl', 0)                
                ops.setCreep(1)
                
                
                ops.wipeAnalysis()
                # Rebuild analysis with new material state
                ops.system('UmfPack')
                ops.test('NormUnbalance', 1e-2, 100)
                ops.algorithm('NewtonLineSearch')
                ops.integrator('LoadControl', 0)
                ops.analysis('Static')

            ops.setTime(t)

            ok = ops.analyze(1)

            record()
            
            results.time_in_longterm_analysis.append(ops.getTime())
            results.deformation_in_longterm_analysis.append(ops_get_maximum_abs_disp(self))
            
            
            # Check for limits
            exit_message = check_analysis_limits(
                    results,
                    eigenvalue_limit=config['eigenvalue_limit'],
                    deformation_limit=config['deformation_limit'],
                    concrete_strain_limit=config['concrete_strain_limit'],
                    steel_strain_limit=config['steel_strain_limit'],
                    section_type=type(self.section).__name__,
                )
            if exit_message:
                results.exit_message = exit_message + f'_while maintaining sustained load at time: {t}'
                break
            
            if ok < 0:
                if t == self.section.Tcr:
                    logger.warning('Analysis failed on the first step of maintaining sustained load')
                    results.exit_message = 'Analysis failed on the first step of maintaining sustained load'
                    return results
                else:
                    logger.warning(f'Analysis failed while maintaining sustained load, time: {t}')
                    results.exit_message = f'Analysis failed while maintaining sustained load: {t}'
                return results

            if config['deformation_limit'] is not None:
                if results.maximum_abs_disp[-1] > config['deformation_limit']:
                    logger.warning(f'Analysis failed while maintaining sustained load: {t}')
                    results.exit_message = 'Deformation Limit Reached'
                    return results
            
            if config['eigenvalue_limit'] is not None:
                if results.lowest_eigenvalue[-1] < config['eigenvalue_limit']:
                    logger.warning(f'Analysis failed while maintaining sustained load: {t}')
                    results.exit_message = 'Eigenvalue Limit Reached'
                    return results

            logt0 = log10(t)
            logt1 = logt0 + 0.01
            t1 = 10 ** logt1
            t = t1
            
        # endregion


        def update_dU(disp_incr_factor, div_factor=1):
            base_incr_factor = config['disp_incr_factor']
            step_scale = disp_incr_factor / base_incr_factor if base_incr_factor else 1.0

            if config['e'] == 0.0:
                # axial-only: shrink load step, stay in LoadControl
                dF = (1.0 / max(1, config['num_steps_vertical'])) * step_scale / div_factor
                ops.integrator('LoadControl', dF)
            else:
                # bending present: DisplacementControl as before
                et, eb = self._oriented_ecc
                if et * eb < 0:
                    dof = 3 * self.ops_n_elem // 4 if abs(et) >= abs(eb) else 1 * self.ops_n_elem // 4
                    dU = self.length * disp_incr_factor / 2 / div_factor
                    ops.integrator('DisplacementControl', dof, 1, dU)
                else:
                    dU = self.length * disp_incr_factor / div_factor
                    ops.integrator('DisplacementControl', self.ops_mid_node, 1, dU)

        def reset_analysis_options(disp_incr_factor):
            update_dU(disp_incr_factor)
            ops.algorithm('Newton')
            ops.test('NormUnbalance', 1e-3, 10)

        # region run final loading phase
        ops.setCreep(0)
        ops.setTime(self.section.Tcr+self.t_sus+1)
        ops.integrator('LoadControl', 0)
        ops.analyze(1)

        record()

        ops.timeSeries('Ramp', 1000, self.section.Tcr+self.t_sus+1, self.section.Tcr+self.t_sus+1, '-smooth', 0.0, '-factor', 100)
        ops.pattern('Plain', 2000, 1000)

        ops.wipeAnalysis()
        if config['e'] == 0:
            dF = self.P_sus / config['num_steps_vertical']
            ops.load(self.ops_n_elem, 0, -1, 0)
            ops.load(0, 0, 0, 0)
            ops.integrator('LoadControl', dF)
        elif sgn_et != sgn_eb:
            if abs(et) >= abs(eb):
                dof = 3 * self.ops_n_elem // 4
            else:
                dof = 1 * self.ops_n_elem // 4
            dU = self.length * config['disp_incr_factor'] / 20
            ops.load(self.ops_n_elem, 0, -1, et * config['e'])
            ops.load(0, 0, 0, -eb * config['e'])
            ops.integrator('DisplacementControl', dof, 1, dU)
        else:
            dU = self.length * config['disp_incr_factor'] / 10
            ops.load(self.ops_n_elem, 0, -1, et * config['e'])
            ops.load(0, 0, 0, -eb * config['e'])
            ops.integrator('DisplacementControl', self.ops_mid_node, 1, dU)

        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.test('EnergyIncr', 1e-4, 100, 0, 2)
        ops.algorithm('ModifiedNewton')
        ops.analysis('Static', '-noWarnings')
        ops.analyze(1)

        maximum_applied_axial_load = 0.
        base_incr_factor = config['disp_incr_factor']
        disp_incr_factor = base_incr_factor
        
        while True:
            ok = ops.analyze(1)
            recovered_div = None

            if ok != 0 and config['try_smaller_steps']:
                for div_factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
                    update_dU(disp_incr_factor, div_factor)
                    ok = ops.analyze(1)
                    if ok != 0:
                        ok = try_analysis_options()
                    if ok == 0:
                        recovered_div = div_factor
                        break

            if ok == 0 and config['try_smaller_steps']:
                disp_incr_factor = adapt_step_factor(disp_incr_factor, base_incr_factor,
                                                     recovered_div)
                reset_analysis_options(disp_incr_factor)

            elif ok != 0:
                results.exit_message = 'Analysis Failed after sustained load phase'
                warnings.warn('Analysis Failed after sustained load phase')
                break

            record()

            # region Check the exit conditions
            # Check for drop in applied load
            if config['percent_load_drop_limit'] is not None:
                current_applied_axial_load = results.applied_axial_load[-1]
                maximum_applied_axial_load = max(maximum_applied_axial_load, current_applied_axial_load)
                if current_applied_axial_load < (1 - config['percent_load_drop_limit']) * maximum_applied_axial_load:
                    results.exit_message = 'Load Drop Limit Reached'
                    break
            
            # Check for limits
            exit_message = check_analysis_limits(
                    results,
                    eigenvalue_limit=config['eigenvalue_limit'],
                    deformation_limit=config['deformation_limit'],
                    concrete_strain_limit=config['concrete_strain_limit'],
                    steel_strain_limit=config['steel_strain_limit'],
                    section_type=type(self.section).__name__,
                )
            if exit_message:
                results.exit_message = exit_message
                break

        # endregion
        return results

    
    def _run_ops_nonproportional_limit_point(self, config, results):
        """Run nonproportional limit point analysis."""
        def update_dU(disp_incr_factor, div_factor=1):
            et, eb = self._oriented_ecc
            if et * eb < 0:
                if abs(et) >= abs(eb):
                    dof = 3 * self.ops_n_elem // 4
                else:
                    dof = 1 * self.ops_n_elem // 4
                dU = self.length * disp_incr_factor / 2 / div_factor
                ops.integrator('DisplacementControl', dof, 1, dU)
            else:
                dU = self.length * disp_incr_factor / div_factor
                ops.integrator('DisplacementControl', self.ops_mid_node, 1, dU)

        def reset_analysis_options(disp_incr_factor):
            update_dU(disp_incr_factor)
            ops.algorithm('Newton')
            ops.test('NormUnbalance', 1e-3, 10)

        # region Run vertical load (time = LFV)
        ops.timeSeries('Linear', 100)
        ops.pattern('Plain', 200, 100)
        ops.load(self.ops_n_elem, 0, -1, 0)
        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.test('NormUnbalance', 1e-3, 10)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', config['P'] / config['num_steps_vertical'])
        ops.analysis('Static', '-noWarnings')
        
        # Define recorder
        def record():
            time = ops.getTime()
            section_strains = ops_get_section_strains(self)

            results.applied_axial_load.append(time)
            results.applied_moment_top.append(0)
            results.applied_moment_bot.append(0)
            results.maximum_abs_moment.append(ops_get_maximum_abs_moment(self))
            results.maximum_abs_disp.append(ops_get_maximum_abs_disp(self))
            results.lowest_eigenvalue.append(ops.eigen('-fullGenLapack', 1)[0])
            if type(self.section).__name__ == "RC":
                results.maximum_concrete_compression_strain.append(section_strains[0])
                results.maximum_steel_strain.append(section_strains[1])
            elif type(self.section).__name__ == "I_shape":
                results.maximum_compression_strain.append(section_strains[0])
                results.maximum_tensile_strain.append(section_strains[1])

            if self.axis == 'x':
                results.curvature.append(section_strains[2])
            elif self.axis == 'y':
                results.curvature.append(section_strains[3])
            else:
                raise ValueError(f'The value of axis ({self.axis}) is not supported.')

        
        record()
        
        for i in range(config['num_steps_vertical']):
            ok = ops.analyze(1)
            
            if ok != 0:
                results.exit_message = 'Analysis Failed In Vertical Loading'
                warnings.warn('Analysis Failed In Vertical Loading')
                return results
            
            record()
            
            # Check for  limits
            exit_message = check_analysis_limits(
                    results,
                    eigenvalue_limit=config['eigenvalue_limit'],
                    deformation_limit=config['deformation_limit'],
                    concrete_strain_limit=config['concrete_strain_limit'],
                    steel_strain_limit=config['steel_strain_limit'],
                    section_type=type(self.section).__name__,
                )
            if exit_message:
                results.exit_message = exit_message
                return results

        # endregion

        # region Run lateral load (time = LFH)
        ops.loadConst('-time', 0.0)
        ops.timeSeries('Linear', 101)
        ops.pattern('Plain', 201, 101)
        
        et, eb = self._oriented_ecc

        if et * eb < 0:
            if abs(et) >= abs(eb):
                dof = 3 * self.ops_n_elem // 4
            else:
                dof = 1 * self.ops_n_elem // 4
            dU = self.length * config['disp_incr_factor'] / 2
            ops.load(self.ops_n_elem, 0, 0, et * config['e'])
            ops.load(0, 0, 0, -eb * config['e'])
            ops.integrator('DisplacementControl', dof, 1, dU)
        else:
            dU = self.length * config['disp_incr_factor']
            ops.load(self.ops_n_elem, 0, 0, et * config['e'])
            ops.load(0, 0, 0, -eb * config['e'])
            ops.integrator('DisplacementControl', self.ops_mid_node, 1, dU)
        
        ops.analysis('Static', '-noWarnings')
        
        # Define recorder
        def record():
            time = ops.getTime()
            section_strains = ops_get_section_strains(self)

            results.applied_axial_load.append(config['P'])
            results.applied_moment_top.append(et * config['e'] * time)
            results.applied_moment_bot.append(-eb * config['e'] * time)
            results.maximum_abs_moment.append(ops_get_maximum_abs_moment(self))
            results.maximum_abs_disp.append(ops_get_maximum_abs_disp(self))
            results.lowest_eigenvalue.append(ops.eigen('-fullGenLapack', 1)[0])
            if type(self.section).__name__ == "RC":
                results.maximum_concrete_compression_strain.append(section_strains[0])
                results.maximum_steel_strain.append(section_strains[1])
            elif type(self.section).__name__ == "I_shape":
                results.maximum_compression_strain.append(section_strains[0])
                results.maximum_tensile_strain.append(section_strains[1])

            if self.axis == 'x':
                results.curvature.append(section_strains[2])
            elif self.axis == 'y':
                results.curvature.append(section_strains[3])
            else:
                raise ValueError(f'The value of axis ({self.axis}) is not supported.')

        record()
        
        maximum_moment = 0
        base_incr_factor = config['disp_incr_factor']
        disp_incr_factor = base_incr_factor

        while True:
            ok = ops.analyze(1)
            recovered_div = None

            if ok != 0 and config['try_smaller_steps']:
                for div_factor in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
                    update_dU(disp_incr_factor, div_factor)
                    ok = ops.analyze(1)
                    if ok != 0:
                        ok = try_analysis_options()
                    if ok == 0:
                        recovered_div = div_factor
                        break

            if ok != 0 and not config['try_smaller_steps']:
                ok = try_analysis_options()

            if ok == 0:
                disp_incr_factor = adapt_step_factor(disp_incr_factor, base_incr_factor,
                                                     recovered_div)
                reset_analysis_options(disp_incr_factor)
            elif ok != 0:
                results.exit_message = 'Analysis Failed'
                warnings.warn('Analysis Failed')
                break
            
            record()

            # Check for drop in applied load (time = the horzontal load factor)
            if config['percent_load_drop_limit'] is not None:
                current_moment = results.maximum_abs_moment[-1]
                maximum_moment = max(current_moment, maximum_moment)
                if current_moment < (1 - config['percent_load_drop_limit']) * maximum_moment:
                    results.exit_message = 'Load Drop Limit Reached'
                    break
                
            # Check for  limits
            exit_message = check_analysis_limits(
                    results,
                    eigenvalue_limit=config['eigenvalue_limit'],
                    deformation_limit=config['deformation_limit'],
                    concrete_strain_limit=config['concrete_strain_limit'],
                    steel_strain_limit=config['steel_strain_limit'],
                    section_type=type(self.section).__name__,
                )
            if exit_message:
                results.exit_message = exit_message
                break
        
        return results
   

    def run_aci_elastic_second_order_analysis(self, analysis_type, **kwargs):
        """
        Run ACI 318 elastic second-order analysis
        
        MODIFIED: Runs until natural limit is reached (no forced targeting)
        
        Parameters:
            analysis_type: 'proportional_limit_point' or 'nonproportional_limit_point'
            **kwargs:
                section_id: Section ID (default 1)
                e: Eccentricity ratio for proportional analysis (M/P)
                P: Axial load for nonproportional analysis
                num_steps_vertical: Number of steps for vertical loading (default 100)
                disp_incr_factor: Displacement increment factor (default 1e-5)
                eigenvalue_limit: Eigenvalue limit for stability (default 0.0)
                deformation_limit: Maximum deformation limit (default 0.1*L)
                EI_type: Name of a load-independent built-in stiffness method
                    ('aci-a', 'aci-b', 'gross'). Load-dependent methods are
                    rejected: the elastic member is built once, before loading,
                    so its stiffness cannot depend on P or M.
                EI: Custom effective stiffness, constant or callable, passed
                    through to RC.EIeff. A callable is evaluated once with
                    P = M = None.
                EI_kwargs: Extra keyword arguments for a callable EI
                betadns: Sustained load ratio used by EI_type (default 0.0)

                With neither EI nor EI_type given, the stiffness falls back to
                0.875*(0.2*Ec*Ig + Es*Isr) for backwards compatibility.

        Returns:
            AnalysisResults object with applied loads and second-order moments
        """
        
        #region Extract parameters
        section_id = kwargs.get('section_id', 1)
        e = kwargs.get('e', 1.0)
        P = kwargs.get('P', 0)
        num_steps_vertical = kwargs.get('num_steps_vertical', 100)
        disp_incr_factor = kwargs.get('disp_incr_factor', 1e-5)
        eigenvalue_limit = kwargs.get('eigenvalue_limit', 0.0)
        deformation_limit = kwargs.get('deformation_limit', 0.1 * self.length)
        max_1_4_Mu_limit = kwargs.get('max_1_4_Mu_limit', True)
        section_factored = kwargs.get('section_factored', False)
        # ACI 318-19 (6.6.4.5.4) covers member out-of-straightness through the
        # minimum moment Mmin = Pu*(0.6 + 0.03h). This routine emulates the code
        # procedure, so it builds the column straight by default rather than
        # also applying self.dxo, which would count the same effect twice. Pass
        # include_imperfection=True to model dxo explicitly instead.
        include_imperfection = kwargs.get('include_imperfection', False)
        # Effective flexural stiffness of the elastic member. The model is built
        # once with a single Elastic section, so the stiffness must be constant
        # over the analysis: EI_type is restricted to the load-independent
        # methods, and a callable EI is evaluated once, before loading.
        EI_type = kwargs.get('EI_type', None)
        EI = kwargs.get('EI', None)
        EI_kwargs = kwargs.get('EI_kwargs', None)
        betadns = kwargs.get('betadns', 0.0)
        #endregion


        #region Create the section's interaction diagram
        P_section, M_section, _ = self.section.section_interaction_2d(
            self.axis, 100, factored=section_factored, only_compressive=True
        )
        section_interaction = InteractionDiagram2d(M_section, P_section, is_closed=False)
        
        P_max_material = np.max(P_section)
        #endregion
        
        
        #region Calculate elastic properties per ACI 318-19
        EI_gross = self.section.EIgross(self.axis)
        if EI is None and EI_type is None:
            # Historical default, kept so existing callers are unaffected:
            # 0.875 times ACI 318-19 (6.6.4.4.4b). Prefer passing EI or EI_type.
            EI_eff = 0.875 * (0.2 * self.section.Ec * self.section.Ig(self.axis)
                              + self.section.Es * self.section.Isr(self.axis))
        else:
            if isinstance(EI_type, str) and EI_type.strip().lower() in _LOAD_DEPENDENT_EI_TYPES:
                raise ValueError(
                    f'EI_type {EI_type!r} depends on the applied load, but this '
                    f'analysis builds one Elastic section before loading begins '
                    f'and so needs a constant stiffness. Pass a constant EI, or '
                    f'a callable EI that does not use P or M.')
            EI_eff = self.section.EIeff(
                self.axis, EI_type, betadns, P=None, M=None, col=self,
                EI=EI, EI_kwargs=EI_kwargs)
            EI_eff = float(EI_eff)
            if not np.isfinite(EI_eff) or EI_eff <= 0:
                raise ValueError(
                    f'Effective stiffness must be finite and positive '
                    f'(got {EI_eff!r})')
        A = self.section.Ag
        E = self.section.Ec
        I_eff = EI_eff / E
        #endregion

        
        #region Build OpenSees model
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        
        # Create nodes, straight unless an explicit imperfection is requested
        for i in range(self.ops_n_elem + 1):
            if not include_imperfection or self.dxo is None:
                x = 0.0
            elif isinstance(self.dxo, (int, float)):
                x = sin(i / self.ops_n_elem * pi) * self.dxo
            else:
                raise ValueError(f'Unknown value of dxo ({self.dxo})')
            y = i / self.ops_n_elem * self.length
            ops.node(i, x, y)
            ops.mass(i, 1, 1, 1)
        
        # Boundary conditions (non-sway column)
        ops.fix(0, 1, 1, 0)  # Pin at bottom (fixed x, y; free rotation)
        ops.fix(self.ops_n_elem, 1, 0, 0)  # Roller at top (fixed x; free y, rotation)
        
        # Geometric transformation (Corotational for large displacement)
        ops.geomTransf(self.ops_geom_transf_type, 100)
        
        # Define ELASTIC section (constant EI per ACI 318)
        ops.section('Elastic', section_id, E, A, I_eff)
        
        # Create elements
        ops.beamIntegration("Lobatto", 1, section_id, self.ops_integration_points)
        for i in range(self.ops_n_elem):
            ops.element('mixedBeamColumn', i, i, i+1, 100, 1)
        #endregion

        
        #region Initialize results
        results = AnalysisResults()
        results.exit_message = ""
        attributes = ['applied_axial_load', 'applied_moment_top', 'applied_moment_bot', 
                    'maximum_abs_moment', 'maximum_abs_disp', 'lowest_eigenvalue']
        for attribute in attributes:
            setattr(results, attribute, [])
        #endregion

        
        #region Determine eccentricity sign
        et, eb = self._oriented_ecc
        #endregion

        
        #region Define recorder function
        def record():
            time = ops.getTime()
            
            # Get maximum moment along column
            max_M = 0
            for i in range(self.ops_n_elem):
                try:
                    forces = ops.eleForce(i)
                    M_i = abs(forces[2])  # Moment at i-end
                    M_j = abs(forces[5])  # Moment at j-end
                    max_M = max(max_M, M_i, M_j)
                except Exception as e:
                    # eleForce might fail if element is not fully formed, skip
                    logger.debug(f'eleForce failed for element {i}, skipping: {e}')

            # Get maximum displacement
            max_disp = 0
            for i in range(self.ops_n_elem + 1):
                disp = abs(ops.nodeDisp(i, 1))
                max_disp = max(max_disp, disp)

            results.maximum_abs_moment.append(max_M)
            results.maximum_abs_disp.append(max_disp)
            try:
                results.lowest_eigenvalue.append(ops.eigen('-fullGenLapack', 1)[0])
            except Exception as e:
                raise RuntimeError('Eigenvalue extraction failed during recording.') from e
        #endregion
        
        
        #region Analysis setup based on type
        if analysis_type.lower() == 'proportional_limit_point':
            # Proportional loading: P and M increase together
            ops.timeSeries('Linear', 100)
            ops.pattern('Plain', 200, 100)
            
            # Applied loads proportional to load factor (time)
            ops.load(self.ops_n_elem, 0, -1, et * e)
            ops.load(0, 0, 0, -eb * e)
            
            ops.constraints('Plain')
            ops.numberer('RCM')
            ops.system('UmfPack')
            ops.test('NormUnbalance', 1e-6, 10)
            ops.algorithm('Newton')
            
            # Determine displacement control node
            if e == 0.0 or (abs(et) < 1e-12 and abs(eb) < 1e-12):
                dF = 1/num_steps_vertical
                ops.integrator('LoadControl', dF)
            else:
                # Determine displacement control node
                if et * eb < 0:
                    if abs(et) >= abs(eb):
                        control_node = 3 * self.ops_n_elem // 4
                    else:
                        control_node = 1 * self.ops_n_elem // 4
                    dU = self.length * disp_incr_factor / 2
                else:
                    control_node = self.ops_n_elem // 2
                    dU = self.length * disp_incr_factor
            
                ops.integrator('DisplacementControl', control_node, 1, dU)

            ops.analysis('Static')
            
            # Recorder for proportional analysis
            def record_prop():
                time = ops.getTime()
                results.applied_axial_load.append(time)
                results.applied_moment_top.append(et * e * time)
                results.applied_moment_bot.append(-eb * e * time)
                record()
            
            record_prop()
            
            # Run analysis
            max_applied_load = 0.0
            M_check = 0.0
            P_check = 0.0
            # Under proportional loading M2/M1 rises monotonically from about
            # 1 + dxo/e, so it can start above the limit and never come back.
            # Recording whether it was ever acceptable separates "amplified past
            # the limit under load" from "over the limit at every load level".
            prop_ratio_has_been_below_limit = False
            
            while True:
                ok = ops.analyze(1)
                
                if ok != 0:
                    # Try alternative algorithms
                    for algo in ['ModifiedNewton', 'KrylovNewton']:
                        ops.algorithm(algo)
                        ok = ops.analyze(1)
                        if ok == 0:
                            break
                    
                    if ok != 0:
                        results.exit_message = 'Analysis Failed - Convergence'
                        warnings.warn('Analysis failed to converge')
                        break
                    else:
                        ops.algorithm('Newton')
                
                record_prop()
                
                # Check for load drop
                current_load = results.applied_axial_load[-1]
                
                if current_load > max_applied_load:
                    max_applied_load = current_load
                
                if current_load < 0.95 * max_applied_load and max_applied_load > 0:
                    results.exit_message = 'Peak Load Reached'
                    break
                
                P_check = results.applied_axial_load[-1]
                M_check = results.maximum_abs_moment[-1]                    
                
                # Check P against P_max_material first
                if P_check > P_max_material:
                    results.exit_message = 'Material Strength Limit Reached'
                    P_check = results.applied_axial_load[-2]
                    M_check = results.maximum_abs_moment[-2]
                    break
                
                # Check M, but only if P and M are positive
                if M_check > 1e-6 and P_check > 1e-6:
                    try:
                        M_boundary = section_interaction.find_x_given_y(P_check, 'pos')
                        logger.debug(f'M_boundary: {M_boundary}, M_check: {M_check}')
                        if M_check > M_boundary:
                            results.exit_message = 'Material Strength Limit Reached'
                            break
                    except Exception as exc:
                        # If find_x_given_y fails, the point is outside the valid range
                        logger.debug(f'find_x_given_y failed, treating point as outside the valid range: {exc}')
                        results.exit_message = 'Material Strength Limit Reached'
                        break

                # Check eigenvalue
                if eigenvalue_limit is not None:
                    if results.lowest_eigenvalue[-1] < eigenvalue_limit:
                        results.exit_message = 'Eigenvalue Limit Reached'
                        break
                
                # ACI 6.2.5.3: flag a column whose second-order moment has
                # amplified past 1.4x its first-order moment. Skipped for
                # axial-only loading, where there is no first-order moment to
                # measure against.
                if max_1_4_Mu_limit is True and not (e == 0 or (abs(et) < 1e-12 and abs(eb) < 1e-12)):
                    first_order = self._applied_first_order_moment(
                        results.applied_moment_top[-1], results.applied_moment_bot[-1])
                    if first_order > 0:
                        if 1.4 * first_order > results.maximum_abs_moment[-1]:
                            prop_ratio_has_been_below_limit = True
                        elif prop_ratio_has_been_below_limit:
                            results.exit_message = 'max_1_4_Mu_limit_reached'
                            break
                        else:
                            results.exit_message = 'max_1_4_Mu_limit_exceeded_at_all_loads'
                            break

                # Check deformation
                if deformation_limit is not None:
                    if results.maximum_abs_disp[-1] > deformation_limit:
                        results.exit_message = 'Deformation Limit Reached'
                        break
        
        elif analysis_type.lower() == 'nonproportional_limit_point':
            # Stage 1: Apply axial load
            ops.timeSeries('Linear', 100)
            ops.pattern('Plain', 200, 100)
            ops.load(self.ops_n_elem, 0, -1, 0)
            
            ops.constraints('Plain')
            ops.numberer('RCM')
            ops.system('UmfPack')
            ops.test('NormUnbalance', 1e-6, 10)
            ops.algorithm('Newton')
            
            # Check for P=0 case
            if P == 0:
                dF = 0
                num_steps_vertical = 1 # Just run one step of P=0
            else:
                dF = P / num_steps_vertical
                
            ops.integrator('LoadControl', dF)
            ops.analysis('Static')
            
            # Recorder for stage 1
            def record_stage1():
                time = ops.getTime()
                results.applied_axial_load.append(time)
                results.applied_moment_top.append(0)
                results.applied_moment_bot.append(0)
                record()
            
            record_stage1()
            
            # Apply axial load
            for i in range(num_steps_vertical):
                ok = ops.analyze(1)
                if ok != 0:
                    results.exit_message = 'Analysis Failed - Vertical Loading'
                    warnings.warn('Analysis failed during vertical loading')
                    break
                
                record_stage1()
                
                # Check if P exceeds material capacity
                P_check = results.applied_axial_load[-1]

                if P_check > P_max_material:
                    results.exit_message = 'Material Strength Limit Reached'
                    break
                
            
            if results.exit_message == "":
                # Stage 2: Apply lateral load while maintaining axial
                ops.loadConst('-time', 0.0)
                ops.timeSeries('Linear', 101)
                ops.pattern('Plain', 201, 101)
                
                time_offset = ops.getTime() 
                
                # Check if axial-only column
                is_axial_only = abs(et) < 1e-10 and abs(eb) < 1e-10
                
                if is_axial_only:
                    # Apply unit moments for axial-only columns
                    ops.load(self.ops_n_elem, 0, 0, 1.0)
                    ops.load(0, 0, 0, -1.0)
                else:
                    # Normal columns with eccentricity
                    ops.load(self.ops_n_elem, 0, 0, et * e)
                    ops.load(0, 0, 0, -eb * e)
                    
                # Determine control node and apply moments
                if et * eb < 0:
                    if abs(et) >= abs(eb):
                        control_node = 3 * self.ops_n_elem // 4
                    else:
                        control_node = 1 * self.ops_n_elem // 4
                    dU = self.length * disp_incr_factor / 2
                else:
                    control_node = self.ops_n_elem // 2
                    dU = self.length * disp_incr_factor
                
                ops.test('NormUnbalance', 1e-6, 100)  # tolerance, max iterations
                ops.algorithm('Newton')
                ops.integrator('DisplacementControl', control_node, 1, dU)
                ops.analysis('Static')
                
                # Recorder for stage 2
                def record_stage2():
                    time = ops.getTime() - time_offset
                    results.applied_axial_load.append(P)
                    if is_axial_only:
                        # For axial-only: moment = time
                        results.applied_moment_top.append(time)
                        results.applied_moment_bot.append(-time)
                    else:
                        # Normal columns
                        results.applied_moment_top.append(et * e * time)
                        results.applied_moment_bot.append(-eb * e * time)
                    record()
                
                record_stage2()
                
                # M2/M1 is not monotonic over this stage. The applied moment
                # starts at zero while M2 already carries P*delta from the
                # axial stage, so the ratio starts arbitrarily large, falls as
                # the applied moment grows, then rises again toward failure.
                # The ACI 6.2.5.3 limit is the crossing on the way back up, so
                # ignore the ratio until it has been below 1.4 at least once.
                ratio_has_been_below_limit = False

                # MODIFIED: Run lateral loading with more permissive limits
                max_moment = 0.0
                max_steps = 10000000  # Prevent infinite loops
                step_count = 0
                M_check = 0.0
                P_check = P

                while step_count < max_steps:
                    ok = ops.analyze(1)
                    step_count += 1
                    
                    if ok != 0:
                        # Try alternative algorithms
                        for algo in ['ModifiedNewton', 'KrylovNewton']:
                            ops.algorithm(algo)
                            ok = ops.analyze(1)
                            if ok == 0:
                                break
                        
                        if ok != 0:
                            results.exit_message = 'Analysis Failed - Lateral Loading'
                            warnings.warn('Analysis failed during lateral loading')
                            break
                        else:
                            ops.algorithm('Newton')
                    
                    record_stage2()
                    
                    # Check for moment drop (natural limit)
                    current_moment = results.maximum_abs_moment[-1]
                    if current_moment > max_moment:
                        max_moment = current_moment

                    if current_moment < 0.95 * max_moment and max_moment > 0:
                        results.exit_message = 'Peak Moment Reached'
                        break

                    M_check = results.maximum_abs_moment[-1]
                    
                    # MODIFIED: More permissive material strength check
                    # Allow 5% overshoot to let analysis find natural peak
                    if M_check > 1e-6 and P_check > 1e-6:
                        try:
                            M_boundary = section_interaction.find_x_given_y(P_check, 'pos')
                            if M_check > M_boundary:
                                results.exit_message = 'Material Strength Limit Reached'
                                break
                        except Exception as exc:
                            # If find_x_given_y fails, the point is outside the valid range
                            logger.debug(f'find_x_given_y failed, treating point as outside the valid range: {exc}')
                            results.exit_message = 'Material Strength Limit Reached'
                            break
                    
                    # Check eigenvalue (natural limit)
                    if eigenvalue_limit is not None:
                        if results.lowest_eigenvalue[-1] < eigenvalue_limit:
                            results.exit_message = 'Eigenvalue Limit Reached'
                            break
                    
                    # Check deformation (natural limit)
                    if deformation_limit is not None:
                        if results.maximum_abs_disp[-1] > deformation_limit:
                            results.exit_message = 'Deformation Limit Reached'
                            break
                    
                    if max_1_4_Mu_limit:
                        # ACI 6.2.5.3: the second-order moment must not exceed
                        # 1.4x the first-order moment.
                        if et == 0 and eb == 0:
                            continue  # no applied moment to measure against
                        applied_M1 = self._applied_first_order_moment(
                            results.applied_moment_top[-1], results.applied_moment_bot[-1])
                        over_limit = M_check >= 1.4 * applied_M1
                        if not over_limit:
                            ratio_has_been_below_limit = True
                        elif ratio_has_been_below_limit:
                            results.exit_message = 'max_1_4_Mu_limit_reached'
                            break
                    
                if step_count >= max_steps:
                    results.exit_message = 'Maximum Steps Reached'
        
        else:
            raise ValueError(f'Unknown analysis type: {analysis_type}')
        #endregion
        
        
        logger.debug(f'Applied axial loads: {results.applied_axial_load}')
        logger.debug(f'Applied top moments: {results.applied_moment_top}')
        logger.debug(f'Applied bottom moments: {results.applied_moment_bot}')
        logger.debug(f'Maximum absolute moments: {results.maximum_abs_moment}')
        logger.debug(f'Maximum absolute displacements: {results.maximum_abs_disp}')
        logger.debug(f'et: {self.et}, eb: {self.eb}, e: {e}, oriented: {(et, eb)}')
        
        
        #region Find limit point
        if not hasattr(results, 'exit_message') or results.exit_message == "":
            results.exit_message = 'Analysis Completed'
        #endregion
        
        
        #region Ensure list is not empty before finding limit point
        if len(results.applied_axial_load) == 0:
            # This should not happen, but as a fallback
            results.applied_axial_load_at_limit_point = 0
            results.applied_moment_top_at_limit_point = 0
            results.applied_moment_bot_at_limit_point = 0
            results.maximum_abs_moment_at_limit_point = 0
            results.maximum_abs_disp_at_limit_point = 0
            return results
        #endregion
        
        
        logger.debug(f'Exit message: {results.exit_message}')

        #region Add logic for new Material Strength limit
        if 'Material Strength' in results.exit_message:
            logger.debug('Finding limit point at Material Strength limit...')
            logger.debug(f'P = {P_check}, M_check = {M_check}')
            try:
                M_check_result = section_interaction.find_x_given_y(P_check, 'pos')
                M_check = M_check_result[0] if isinstance(M_check_result, list) else M_check_result
            except Exception as exc:
                if e==0 or (abs(et) < 1e-12 and abs(eb) < 1e-12):
                    logger.debug(f'find_x_given_y failed for axial-only case, defaulting M_check to 0: {exc}')
                    M_check = 0.0
                else:
                    raise RuntimeError('Failed to find M boundary for Material Strength limit point.') from exc
            logger.debug(f'At material strength limit: P = {P_check}, M_boundary = {M_check}')
            ind, x = find_limit_point_in_list(results.maximum_abs_moment, M_check)
            if x == 0:
                ind, x = find_limit_point_in_list(results.applied_axial_load, P_check)
        elif 'Peak Load' in results.exit_message or 'Peak Moment' in results.exit_message:
            ind = len(results.applied_axial_load) - 2 # Go back to the peak
            x = 0.0
        elif 'Eigenvalue' in results.exit_message:
            ind, x = find_limit_point_in_list(results.lowest_eigenvalue, eigenvalue_limit)
        elif 'Deformation' in results.exit_message:
            ind, x = find_limit_point_in_list(results.maximum_abs_disp, deformation_limit)
        elif 'max_1_4_Mu_limit' in results.exit_message:
            ind = len(results.applied_axial_load) - 2 # Go back to the point before the last
            x = 0.0
        else:
            ind = len(results.applied_axial_load) - 1
            x = 0.0
        #endregion
        
        
        #region Safety check for None values
        if ind is None or x is None:
            ind = len(results.applied_axial_load) - 1
            x = 0.0
        #endregion
        
        
        #region Interpolate results at limit point
        results.applied_axial_load_at_limit_point = interpolate_list(results.applied_axial_load, ind, x)
        results.applied_moment_top_at_limit_point = interpolate_list(results.applied_moment_top, ind, x)
        results.applied_moment_bot_at_limit_point = interpolate_list(results.applied_moment_bot, ind, x)
        results.maximum_abs_moment_at_limit_point = interpolate_list(results.maximum_abs_moment, ind, x)
        results.maximum_abs_disp_at_limit_point = interpolate_list(results.maximum_abs_disp, ind, x)
        #endregion
        
        return results
    
    
    def run_aci_elastic_interaction(self, **kwargs):
        """
        Run ACI elastic second-order interaction diagram
        
        MODIFIED: Simplified - no optimization, just run once per point
        
        Parameters:
            **kwargs:
                section_id: Section ID (default 1)
                num_points: Number of points on interaction diagram (default 10)
                prop_disp_incr_factor: Displacement increment for proportional analysis (default 1e-6)
                nonprop_disp_incr_factor: Displacement increment for nonproportional analysis (default 1e-5)
                e: Eccentricity ratio for nonproportional analyses (default 1.0)
                
        Returns:
            Dictionary with keys:
                'P': Array of axial loads
                'M1': Array of first-order moments
                'M2': Array of second-order moments
                'exit_message': List of exit messages for each point

        Additional keyword arguments:
            max_1_4_Mu_limit: Enforce the ACI 318-19 (6.2.5.3) limit that the
                              second-order moment not exceed 1.4 times the
                              first-order moment. Default True.

        The ACI 318-19 (6.6.4.5.4) minimum eccentricity is applied to the
        loading, by constructing the column with apply_minimum_eccentricity.
        """
        
        section_id = kwargs.get('section_id', 1)
        num_points = kwargs.get('num_points', 10)
        prop_disp_incr_factor = kwargs.get('prop_disp_incr_factor', 1e-6)
        nonprop_disp_incr_factor = kwargs.get('nonprop_disp_incr_factor', 1e-5)
        section_factored = kwargs.get('section_factored', False)
        max_1_4_Mu_limit = kwargs.get('max_1_4_Mu_limit', True)
        e = kwargs.get('e', 1.0)
        # 1. Get Elastic Buckling Load (P_cr_elastic)
        logger.debug('Running proportional analysis for elastic buckling load...')
        results_pcr = self.run_aci_elastic_second_order_analysis(
            'proportional_limit_point',
            e=0,
            section_id=section_id,
            section_factored=section_factored,
            max_1_4_Mu_limit=max_1_4_Mu_limit,
            disp_incr_factor=prop_disp_incr_factor)

        P_cr_elastic = results_pcr.applied_axial_load_at_limit_point
        logger.debug(f'  P_cr_elastic = {P_cr_elastic:.2f}')

        # 2. Get Material P-M Diagram (for P_max_material and M_n_material)
        P_sect, M_sect, _ = self.section.section_interaction_2d(self.axis, 100, factored=section_factored,
                                                                only_compressive=True)
        P_max_material = np.max(P_sect)

        # Find pure bending strength (Mn)
        idx_zero = np.argmin(np.abs(P_sect))
        M_n_material = M_sect[idx_zero]
        logger.debug(f'  P_max_material = {P_max_material:.2f}')
        logger.debug(f'  M_n_material = {M_n_material:.2f}')

        # 3. Determine the starting P (the lower of the two, with a small buffer to stay inside the interaction diagram)
        P_start = min(P_cr_elastic, P_max_material) * 0.999  # 0.1% buffer
        logger.debug(f'Starting axial load for interaction diagram: P_start = {P_start:.2f}')

        # Initialize results list with the first point (axial-only loading)
        P = [P_start]
        M1 = [0]
        M2 = [results_pcr.maximum_abs_moment_at_limit_point]
        exit_message = [results_pcr.exit_message]

        if np.isnan(P[0]):
            raise ValueError('Analysis failed at axial-only loading')

        # 4. Loop through axial loads - SIMPLIFIED: No optimization
        logger.debug(f'Generating {num_points} points on interaction diagram...')
        for i in range(1, num_points):
            # Iterate from P_start down to 0
            iP = P_start * (num_points - 1 - i) / (num_points - 1)

            if iP < 1e-6:
                # Pure bending case (P ≈ 0)
                P.append(0)
                M1.append(M_n_material)
                M2.append(M_n_material)
                exit_message.append('Pure Bending (Material Strength)')
            else:
                # Run the analysis ONCE - no iteration
                results = self.run_aci_elastic_second_order_analysis(
                    'nonproportional_limit_point',
                    P=iP,
                    e=e,
                    section_id=section_id,
                    section_factored=section_factored,
                    max_1_4_Mu_limit=max_1_4_Mu_limit,
                    disp_incr_factor=nonprop_disp_incr_factor
                )

                # Record whatever we got at the limit point
                P.append(iP)
                M1.append(self._applied_first_order_moment(
                    results.applied_moment_top_at_limit_point,
                    results.applied_moment_bot_at_limit_point))
                M2.append(results.maximum_abs_moment_at_limit_point)
                exit_message.append(results.exit_message)

        logger.debug('Interaction diagram generation complete!')

        return {
            'P': np.array(P),
            'M1': np.array(M1),
            'M2': np.array(M2),
            'exit_message': exit_message
        }