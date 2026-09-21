import warnings
from math import sqrt, pi, ceil, exp, sin, log10
import matplotlib.pyplot as plt
import numpy as np
from libdenavit import opensees as ops
from libdenavit.OpenSees import circ_patch_2d, obround_patch_2d, obround_patch_2d_confined
from libdenavit.section import AciStrainCompatibility, FiberSection, ACI_phi


class RC:
    _default_age_warn = False
    def __init__(self, conc_cross_section, reinforcement, fc, fy, units, dbt=None, s=None, fyt=None, lat_config="A",
                 transverse_reinf_type='ties', tD=5, Tcr=28, tcast=0):
        self._treat_reinforcement_as_point = True
        self._Ec = None
        self._Es = None
        self._eps_c = None
        self._Abt = None
        self._Mn = dict()
        self._CS_id2d = dict()
        self._reinforcement = None

        self.conc_cross_section = conc_cross_section
        self.reinforcement = reinforcement
        self.fc = fc
        self.fy = fy
        self.units = units.lower()
        self.dbt = dbt
        self.s = s
        self.fyt = fyt
        self.lat_config = lat_config
        self.transverse_reinf_type = transverse_reinf_type
        self.tD = tD
        self.Tcr = Tcr
        self.tcast = tcast
        self.has_concrete = True
        self.has_steel = True



    @staticmethod
    def _require_positive(name, value):
        """Reject a non-finite or non-positive material property.

        A negative modulus propagates silently into a negative EIgross, and
        a negative fc reaches sqrt() as a bare math domain error.
        """
        if value is None:
            raise ValueError(f'{name} must be set to a finite positive value')
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be numeric (got {value!r})') from exc
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(f'{name} must be a finite positive value (got {value!r})')
        return numeric

    @property
    def Ec(self):
        if self._Ec is not None:
            return self._Ec

        self._require_positive('fc', self.fc)
        if self.units == 'us':
            return 57 * sqrt(self.fc * 1000)

        if self.units == 'si':
            return 4700 * sqrt(self.fc)

        raise ValueError(f'Ec is not set and default is not implemented for {self.units = }')
    @Ec.setter
    def Ec(self, x):
        if x is not None:
            self._require_positive('Ec', x)
        self._invalidate_section_caches()
        self._Ec = x

    def get_shrinkage_props_for_uniaxial_material(self, print_factors=False, **kwargs):
        """
        ACI 209R-92 shrinkage properties for the OpenSees CreepShrinkageACI209 material.

        Input conventions, which the correction equations themselves fix:
            RH              ambient relative humidity as a fraction, 0.4 to 1.0
            fine_agg_ratio  fine aggregate by weight, in percent (50 means 50%)
            air_content     air content in percentage points (6 means 6%)
            VoverS          volume-to-surface ratio in length units
            t0 / tc         age at loading / curing duration, days
        """
        # Define default values for 'si' and 'us' units
        default_values = {'si':
                              {'eps_sh_u0': 780e-6,  # ultimate shrinkage strain
                               'tc': 7,  # initial moist curing time
                               'RH': 0.4,  # ambient relative humidity
                               'fine_agg_ratio': 50,  # fine aggregate ratio
                               'air_content': 6,  # air content, percentage points
                               'VoverS': 38,  # volume to surface ratio
                               'slump': 70,  # slump (mm)
                               'cement_content': 360  # cement content (kg/m^3)
                               },
                          'us':
                              {'eps_sh_u0': 780e-6,  # ultimate shrinkage strain
                               'tc': 7,  # initial moist curing time
                               'RH': 0.4,  # ambient relative humidity
                               'fine_agg_ratio': 50,  # fine aggregate ratio
                               'air_content': 6,  # air content, percentage points
                               'VoverS': 1.5,  # volume to surface ratio
                               'slump': 2.7,  # slump (in)
                               'cement_content': 611  # cement content (lb/yd^3)
                               }
                          }

        # Set default values for missing kwargs
        for key, value in default_values[self.units].items():
            if key == 'VoverS':
                continue
            kwargs.setdefault(key, value)

        # Extract parameters from kwargs
        eps_sh_u0 = kwargs['eps_sh_u0']
        tc = kwargs['tc']
        RH = kwargs['RH']
        fine_agg_ratio = kwargs['fine_agg_ratio']
        air = kwargs['air_content']

        try:
            VoverS = kwargs['VoverS']
        except KeyError:
            VoverS = self.conc_cross_section.A / self.conc_cross_section.perimeter

        slump = kwargs['slump']
        cement_content = kwargs['cement_content']

        # Correction for curing time
        if tc == default_values[self.units]['tc']:
            gamma_sh_tc = 1.0
        else:
            if tc <= 0:
                raise ValueError("Curing time 'tc' must be positive.")
            gamma_sh_tc = 1.202 - 0.2337 * log10(tc)

        # Correction for relative humidity
        if RH == default_values[self.units]['RH']:
            gamma_sh_RH = 1.0
        elif 0.4 <= RH <= 0.8:
            gamma_sh_RH = 1.4 - 1.02 * RH
        elif 0.8 < RH <= 1.0:
            gamma_sh_RH = 3.0 - 3.0 * RH
        else:
            raise ValueError("Relative humidity 'RH' must be between 0.4 and 1.")

        # Correction for member size (VoverS)
        if VoverS == default_values[self.units]['VoverS']:
            gamma_sh_VS = 1.0
        elif self.units == 'si':
            gamma_sh_VS = 1.2 * exp(-0.00472 * VoverS)
        elif self.units == 'us':
            gamma_sh_VS = 1.2 * exp(-0.12 * VoverS)

        # Correction for slump
        if slump == default_values[self.units]['slump']:
            gamma_sh_s = 1.0
        elif self.units == 'si':
            gamma_sh_s = 0.89 + 0.00161 * slump
        elif self.units == 'us':
            gamma_sh_s = 0.89 + 0.041 * slump

        # Correction for fine aggregate content
        if fine_agg_ratio == default_values[self.units]['fine_agg_ratio']:
            gamma_sh_psi = 1.0
        elif fine_agg_ratio <= 50:
            gamma_sh_psi = 0.3 + 0.014 * fine_agg_ratio
        else:
            gamma_sh_psi = 0.9 + 0.002 * fine_agg_ratio

        # Correction for cement content
        if cement_content == default_values[self.units]['cement_content']:
            gamma_sh_c = 1.0
        elif self.units == 'si':
            gamma_sh_c = 0.75 + 0.00061 * cement_content
        elif self.units == 'us':
            gamma_sh_c = 0.75 + 0.00036 * cement_content

        # Correction for air content
        if air == default_values[self.units]['air_content']:
            gamma_sh_a = 1.0
        else:
            gamma_sh_a = max(1.0, 0.95 + 0.008 * air)

        # minimum gamma_sh
        min_gamma_sh = 0.2

        # Global correction for ultimate shrinkage strain
        gamma_sh = gamma_sh_tc * gamma_sh_RH * gamma_sh_VS * gamma_sh_s * gamma_sh_psi * gamma_sh_c * gamma_sh_a
        if print_factors:
            print(
                f'gamma_sh = {gamma_sh}, gamma_sh_tc = {gamma_sh_tc}, gamma_sh_RH = {gamma_sh_RH}, gamma_sh_VS = {gamma_sh_VS}, gamma_sh_s = {gamma_sh_s}, gamma_sh_psi = {gamma_sh_psi}, gamma_sh_c = {gamma_sh_c}, gamma_sh_a = {gamma_sh_a}')
        gamma_sh = max(gamma_sh, min_gamma_sh)

        if self.units == 'si':
            f = 26 * exp(0.0142 * VoverS)
        elif self.units == 'us':
            f = 26 * exp(0.36 * VoverS)

        eps_sh_u = -eps_sh_u0 * gamma_sh
        return {'eps_sh_u': eps_sh_u, 'f': f, 'psish': f}

    def get_creep_props_for_uniaxial_material(self, print_factors=False, **kwargs):
        """
        ACI 209R-92 creep properties for the OpenSees CreepShrinkageACI209 material.

        Input conventions, which the correction equations themselves fix:
            RH              ambient relative humidity as a fraction, 0.4 to 1.0
            fine_agg_ratio  fine aggregate by weight, in percent (50 means 50%)
            air_content     air content in percentage points (6 means 6%)
            VoverS          volume-to-surface ratio in length units
            t0 / tc         age at loading / curing duration, days
        """
        # Define default values for 'si' and 'us' units
        default_values = {'si':
                              {'phi_u_0': 2.35,  # ultimate creep coefficient
                               't0': 7,  # age at loading
                               'RH': 0.4,  # ambient relative humidity
                               'VoverS': 38,  # volume to surface ratio
                               'slump': 70,  # slump (mm)
                               'fine_agg_ratio': 50,  # fine aggregate ratio, percent by weight
                               'air_content': 6,  # air content, percentage points
                               },
                          'us':
                              {'phi_u_0': 2.35,  # ultimate creep coefficient
                               't0': 7,  # age at loading
                               'RH': 0.4,  # ambient relative humidity
                               'VoverS': 1.5,  # volume to surface ratio
                               'slump': 2.7,  # slump (mm)
                               'fine_agg_ratio': 50,  # fine aggregate ratio, percent by weight
                               'air_content': 6,  # air content, percentage points
                               }
                          }

        # Set default values
        for key, value in default_values[self.units].items():
            if key == 'VoverS':
                continue
            kwargs.setdefault(key, value)

        # Extract parameters from kwargs
        phi_u_0 = kwargs['phi_u_0']
        t0 = kwargs['t0']
        RH = kwargs['RH']
        try:
            VoverS = kwargs['VoverS']
        except KeyError:
            VoverS = self.conc_cross_section.A / self.conc_cross_section.perimeter
        slump = kwargs['slump']
        fine_agg_ratio = kwargs['fine_agg_ratio']
        air = kwargs['air_content']

        # Correction for curing time
        if 0 <= t0 <= 7:
            gamma_c_t0 = 1.0
        elif t0 > 0:
            gamma_c_t0 = 1.25 * t0 ** -0.118
        else:
            raise ValueError("Loading time 't0' must be positive.")

        # Correction for relative humidity
        if RH < 0.4:
            gamma_c_RH = 1.0
        else:
            gamma_c_RH = 1.27 - 0.67 * RH

        # Correction for member size (VoverS)
        if VoverS == default_values[self.units]['VoverS']:
            gamma_c_VS = 1.0
        elif self.units == 'si':
            gamma_c_VS = 2 / 3 * (1 + 1.13 * exp(-0.0213 * VoverS))
        elif self.units == 'us':
            gamma_c_VS = 2 / 3 * (1 + 1.13 * exp(-0.54 * VoverS))

        # Correction for slump
        if slump == default_values[self.units]['slump']:
            gamma_c_s = 1.0
        elif self.units == 'si':
            gamma_c_s = 0.82 + 0.00264 * slump
        elif self.units == 'us':
            gamma_c_s = 0.82 + 0.067 * slump

        # Correction for fine aggregate content
        if fine_agg_ratio == default_values[self.units]['fine_agg_ratio']:
            gamma_c_psi = 1.0
        else:
            gamma_c_psi = 0.88 + 0.0024 * fine_agg_ratio

        # Correction for air content
        if air == default_values[self.units]['air_content']:
            gamma_c_a = 1.0
        else:
            gamma_c_a = max(1.0, 0.46 + 0.09 * air)

        # Global correction for ultimate shrinkage strain
        gamma_c = gamma_c_t0 * gamma_c_RH * gamma_c_VS * gamma_c_s * gamma_c_psi * gamma_c_a
        if print_factors:
            print(f'gamma_c = {gamma_c}, gamma_c_t0 = {gamma_c_t0}, gamma_c_RH = {gamma_c_RH}, gamma_c_VS = {gamma_c_VS},'
                  f'gamma_c_s = {gamma_c_s}, gamma_c_psi = {gamma_c_psi}, gamma_c_a = {gamma_c_a}')

        phi_u = gamma_c * phi_u_0

        psi = kwargs.get('psi', 1.0)
        if self.units == 'si':
            d = 26 * exp(0.0142 * VoverS)
        elif self.units == 'us':
            d = 26 * exp(0.36 * VoverS)

        return {'phi_u': phi_u, 'psi': psi, "d": d, 'psicr1': psi, 'psicr2': d}

    @property
    def Abt(self):
        if self._Abt is not None:
            return self._Abt
        else:
            return pi * self.dbt ** 2 / 4
    @Abt.setter
    def Abt(self, x):
        self._Abt = x

    @property
    def Es(self):
        if self._Es is not None:
            return self._Es

        if self.units == 'us':
            return 29000

        if self.units == 'si':
            return 2e5

        raise ValueError(f'Es is not set and default is not implemented for {self.units = }')
    @Es.setter
    def Es(self, x):
        if x is not None:
            self._require_positive('Es', x)
        self._invalidate_section_caches()
        self._Es = x

    @property
    def eps_c(self):
        if self._eps_c is not None:
            return self._eps_c
        if self.units == 'us':
            return (self.fc * 1000) ** (1 / 4) / 4000
        if self.units == 'si':
            return (self.fc * 145.038) ** (1 / 4) / 4000

        raise ValueError(f'eps_c is not set and default is not implemented for {self.units = }')
    @eps_c.setter
    def eps_c(self, x):
        if x is not None:
            self._require_positive('eps_c', x)
        self._invalidate_section_caches()
        self._eps_c = x
        

    @property
    def reinforcement(self):
        return self._reinforcement
    @reinforcement.setter
    def reinforcement(self, x):
        """
        Store the reinforcement as a list of patterns.

        A single pattern object becomes a one-item list; a list or tuple is
        copied into a new list.
        """
        if isinstance(x, (list, tuple)):
            patterns = list(x)
        else:
            patterns = [x]

        if not patterns:
            # Several methods (validate_section_reinf, maximum_tensile_steel_strain)
            # index reinforcement[0], so concrete-only sections are not supported.
            raise ValueError('reinforcement must contain at least one reinforcement pattern')

        self._reinforcement = patterns
        self._invalidate_section_caches()

    def depth(self, axis):
        return self.conc_cross_section.depth(axis)

    @property
    def Ag(self):
        return self.conc_cross_section.A

    @property
    def Ac(self):
        return self.Ag - self.Asr

    @property
    def Asr(self):
        a = 0
        for i in self.reinforcement:
            a += i.num_bars * i.Ab
        return a

    def maximum_concrete_compression_strain(self, axial_strain, curvatureX=0, curvatureY=0):
        if type(self.conc_cross_section).__name__ == 'Rectangle':
            extreme_strain = axial_strain - self.conc_cross_section.H/2 * abs(curvatureX) \
                             - self.conc_cross_section.B/2 * abs(curvatureY)
            return extreme_strain

        if type(self.conc_cross_section).__name__ == 'Circle':
            extreme_strain = axial_strain - self.conc_cross_section.diameter/2 * sqrt(curvatureX**2 + curvatureY**2)
            return extreme_strain

        if type(self.conc_cross_section).__name__ == 'Obround':
            a = self.conc_cross_section.a
            D = self.conc_cross_section.D
            extreme_strain = axial_strain \
                - (a / 2) * abs(curvatureY) \
                - (D / 2) * sqrt(curvatureX ** 2 + curvatureY ** 2)
            return extreme_strain

        raise ValueError(f'No maximum concrete compression strain implemented for {type(self.conc_cross_section).__name__ = }')

    def maximum_tensile_steel_strain(self, axial_strain, curvatureX=0, curvatureY=0):
        max_strain = float('-inf')
        for pattern in self.reinforcement:
            x_coordinates, y_coordinates = pattern.coordinates
            for x, y in zip(x_coordinates, y_coordinates):
                strain = axial_strain - y * curvatureX - x * curvatureY
                if strain > max_strain:
                    max_strain = strain
        if max_strain == float('-inf'):
            raise ValueError('No reinforcing bars found to compute a maximum tensile steel strain')
        return max_strain

    def maximum_tensile_strain(self, axial_strain, curvatureX=0, curvatureY=0):
        """Alias for maximum_tensile_steel_strain, so generic code can call a
        common method name across section types without needing to know
        which material-specific name a given section class uses."""
        return self.maximum_tensile_steel_strain(axial_strain, curvatureX, curvatureY)

    def Ig(self, axis):
        return self.conc_cross_section.I(axis)

    def Ic(self, axis):
        return self.Ig(axis) - self.Isr(axis)

    def Isr(self, axis):
        i = 0
        for j in self.reinforcement:
            i += j.I(axis)
        return i

    @property
    def p0(self):
        # Nominal axial strength of section
        self._require_positive('fc', self.fc)
        self._require_positive('fy', self.fy)
        p0 = 0.85 * self.fc * (self.Ag - self.Asr) + self.fy * self.Asr
        return p0

    @property
    def p0g(self):
        # Nominal axial strength of gross concrete section
        p0g = 0.85 * self.fc * self.Ag
        return p0g

    @property
    def pnco(self):
        # See Section 22.4.2 of ACI318-19
        if self.transverse_reinf_type.lower() == 'ties':
            pnco = 0.80 * self.p0
        elif self.transverse_reinf_type.lower() in ['spiral', 'spirals']:
            pnco = 0.85 * self.p0
        else:
            raise ValueError("Unknown transverse_reinf_type")
        return pnco

    def EIgross(self, axis):
        return self.Ec * self.Ic(axis) + self.Es * self.Isr(axis)
    
    # Built-in effective stiffness methods.
    _BUILTIN_EI_TYPES = ('aci-a', 'aci-b', 'aci-c', 'jf-a', 'jf-b', 'gross',
                         'proposed_1_1', 'proposed_1_2', 'proposed_2')

    # Keyword arguments a custom EI callable always receives.
    _EI_RESERVED_KWARGS = ('section', 'axis', 'betadns', 'P', 'M', 'col')

    @staticmethod
    def _as_load_arrays(EI_type, P, M, require_M=True):
        """Normalize P and M to float arrays; say whether the input was scalar."""
        if P is None:
            raise ValueError(f"P must be defined for EI_type = {EI_type!r}")
        if require_M and M is None:
            raise ValueError(f"M must be defined for EI_type = {EI_type!r}")

        try:
            P_array = np.asarray(P, dtype=float)
            M_array = np.asarray(0.0 if M is None else M, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'P and M must be numeric for EI_type = {EI_type!r} '
                f'(got P={P!r}, M={M!r})') from exc

        try:
            shape = np.broadcast_shapes(P_array.shape, M_array.shape)
        except ValueError as exc:
            raise ValueError(
                f'P and M must have compatible shapes for EI_type = {EI_type!r} '
                f'(got {P_array.shape} and {M_array.shape})') from exc

        scalar_input = shape == ()
        P_array = np.broadcast_to(P_array, shape)
        M_array = np.broadcast_to(M_array, shape)
        return np.atleast_1d(P_array), np.atleast_1d(M_array), scalar_input

    @staticmethod
    def _eccentricity_ratio(P, M, depth):
        """abs(M) / abs(P) / depth.

        P == 0 gives infinity, which pushes the method to its lower bound.
        P == 0 with M == 0 gives NaN.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.abs(M) / np.abs(P) / depth

    @staticmethod
    def _match_load_shape(value, scalar_input):
        """Return a scalar for scalar input, otherwise the array unchanged."""
        return float(value[0]) if scalar_input else np.asarray(value)

    def _custom_EIeff(self, custom, axis, betadns, P, M, col, EI_kwargs):
        """Evaluate a user supplied constant or callable effective stiffness."""
        EI_kwargs = dict(EI_kwargs or {})

        if callable(custom):
            clashes = sorted(set(EI_kwargs) & set(self._EI_RESERVED_KWARGS))
            if clashes:
                raise ValueError(
                    f'EI_kwargs may not override the reserved arguments '
                    f'{clashes}; they are always supplied by EIeff.')
            value = custom(section=self, axis=axis, betadns=betadns,
                           P=P, M=M, col=col, **EI_kwargs)
        else:
            if EI_kwargs:
                raise ValueError(
                    'EI_kwargs is only meaningful when the custom EI is callable; '
                    f'got a {type(custom).__name__}.')
            value = custom

        try:
            value = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Custom EI must be numeric, got {value!r}') from exc

        if value.size == 0:
            raise ValueError('Custom EI must not be empty')
        if not np.all(np.isfinite(value)):
            raise ValueError(f'Custom EI must be finite, got {value!r}')
        if np.any(value <= 0):
            raise ValueError(f'Custom EI must be positive, got {value!r}')

        # Match the shape implied by the loads, if any were supplied.
        load_shape = ()
        for load in (P, M):
            if load is not None:
                load_shape = np.broadcast_shapes(
                    load_shape, np.asarray(load, dtype=float).shape)

        if value.ndim == 0:
            if load_shape == ():
                return float(value)
            return np.broadcast_to(value, load_shape).copy()

        if load_shape not in ((), value.shape):
            raise ValueError(
                f'Custom EI shape {value.shape} does not match the load shape '
                f'{load_shape}')
        return value

    def EIeff(self, axis, EI_type=None, betadns=0.0, P=None, M=None, col=None,
              EI=None, EI_kwargs=None):
        """Effective flexural stiffness.

        Parameters
        ----------
        axis : str
            'x' or 'y'.
        EI_type : str
            One of: aci-a, aci-b, aci-c, jf-a, jf-b, gross, proposed_1_1,
            proposed_1_2, proposed_2. Case and spaces are ignored. A
            '<module>,<arg>' string resolves a function from that module. A
            non-string value is treated as a custom EI.
        betadns : float
            Sustained load ratio. Must be greater than -1.
        P, M : scalar or array-like, optional
            Axial load and moment, required by the load-dependent methods.
            Scalar in, scalar out; array in, array out.
        col : object, optional
            Column, required by proposed_2.
        EI : float or callable, optional
            Custom effective stiffness. A constant is used directly. A callable
            is invoked with the reserved keyword arguments section, axis,
            betadns, P, M and col, plus anything in EI_kwargs, and must return
            a finite positive stiffness.
        EI_kwargs : dict, optional
            Extra keyword arguments for a callable EI. Rejected for a constant
            EI, and may not shadow the reserved names.

        Notes
        -----
        P == 0 with M != 0 returns the method lower bound. P == 0 with
        M == 0 returns NaN.
        """
        # --- custom effective stiffness -----------------------------------
        custom = None
        if EI is not None and EI_type is not None and not isinstance(EI_type, str):
            raise ValueError(
                'Provide a custom EI through either EI or a non-string EI_type, '
                'not both.')
        if EI is not None:
            custom = EI
        elif EI_type is not None and not isinstance(EI_type, str):
            custom = EI_type

        if custom is not None:
            return self._custom_EIeff(custom, axis, betadns, P, M, col, EI_kwargs)

        if EI_type is None:
            raise ValueError('EI_type or EI must be provided')

        if EI_kwargs:
            raise ValueError('EI_kwargs is only meaningful together with a custom EI')

        # --- shared validation --------------------------------------------
        if 1 + betadns <= 0:
            raise ValueError(
                f'betadns must be greater than -1 because 1 + betadns divides the '
                f'effective stiffness (got {betadns!r})')

        key = EI_type.strip().lower()
        Ig = self.Ig(axis)
        Ec = self.Ec
        creep_factor = 1 + betadns

        # --- load independent methods -------------------------------------
        if key == 'aci-a':
            # ACI 318-19, Section 6.6.4.4.4a
            return 0.4 * Ec * Ig / creep_factor

        if key == 'aci-b':
            # ACI 318-19, Section 6.6.4.4.4b
            return (0.2 * Ec * Ig + self.Es * self.Isr(axis)) / creep_factor

        if key == 'gross':
            return self.EIgross(axis) / creep_factor

        # --- load dependent methods ---------------------------------------
        if key == 'aci-c':
            # ACI 318-19, Section 6.6.4.4.4c
            P_arr, M_arr, scalar_input = self._as_load_arrays(EI_type, P, M)
            max_I = 0.875 * Ig
            min_I = 0.35 * Ig
            # As P -> 0 the clip returns min_I, the pure-bending limit.
            with np.errstate(divide='ignore', invalid='ignore'):
                I_ACI = (0.8 + 25 * self.Asr / self.Ag) * (
                        1 - np.abs(M_arr) / np.abs(P_arr) / self.depth(axis)
                        - 0.5 * np.abs(P_arr) / self.p0) * Ig
            Ieff = np.clip(I_ACI, min_I, max_I)
            return self._match_load_shape(Ieff * Ec / creep_factor, scalar_input)

        if key == 'jf-a':
            # Jenkins and Frosch, 2011 - Eq.10.1
            P_arr, M_arr, scalar_input = self._as_load_arrays(EI_type, P, M)
            ecc_ratio = self._eccentricity_ratio(P_arr, M_arr, self.depth(axis))
            shape_factor = np.where(ecc_ratio <= 0.1, 1.0, 1.2 - 2 * ecc_ratio)
            EI_jenkins = ((1.05 - 0.6 * np.abs(P_arr) / self.p0)
                          * (1 + 3 * (self.Asr / self.Ag - 0.01))
                          * shape_factor * Ec * Ig / creep_factor)
            min_EI = 0.3 * Ec * Ig / creep_factor
            return self._match_load_shape(np.maximum(EI_jenkins, min_EI), scalar_input)

        if key == 'jf-b':
            # Jenkins and Frosch, 2011 - Eq.10.2
            # Eccentricity ratio uses magnitudes, as in JF-a.
            P_arr, M_arr, scalar_input = self._as_load_arrays(EI_type, P, M)
            ecc_ratio = self._eccentricity_ratio(P_arr, M_arr, self.depth(axis))
            shape_factor = np.where(ecc_ratio <= 0.1, 1.0, 1.2 - 2 * ecc_ratio)
            EI_jenkins = ((1.0 - 0.5 * np.abs(P_arr) / self.p0g)
                          * Ec * Ig * shape_factor / creep_factor)
            min_EI = 0.4 * Ec * Ig / creep_factor
            return self._match_load_shape(np.maximum(EI_jenkins, min_EI), scalar_input)

        if key in ('proposed_1_1', 'proposed_1_2'):
            _, M_arr, scalar_input = self._as_load_arrays(EI_type, 0.0, M)
            Mn = self.Mn(axis)
            if key == 'proposed_1_1':
                low = 0.4 * Ec * Ig / creep_factor
            else:
                low = (0.2 * Ec * Ig + self.Es * self.Isr(axis)) / creep_factor
            high = 1 * self.Es * self.Isr(axis) / creep_factor
            EI_values = np.where(M_arr / Mn <= 0.95, low, high)
            return self._match_load_shape(EI_values, scalar_input)

        if key == 'proposed_2':
            if col is None:
                raise ValueError("col must be defined for EI_type = 'proposed_2'")
            P_arr, _, scalar_input = self._as_load_arrays(EI_type, P, M, require_M=False)
            EI_gross = Ec * Ig
            P0 = self.p0
            r = np.sqrt(Ig / self.Ag)
            L = col.length

            if type(col).__name__ == 'NonSwayColumn2d':
                K = 1
            elif type(col).__name__ == 'SwayColumn2d':
                K = col.effective_length_factor(0.4 * EI_gross)
            else:
                raise ValueError(
                    f"EI_type 'proposed_2' does not support column type "
                    f'{type(col).__name__}')

            EI_values = ((0.45 * P_arr / P0
                          + 0.35 * ((K * L / r) / 100) ** 1.85
                          * np.sin(np.pi * P_arr / P0)) * EI_gross
                         + 0.3 * self.Es * self.Isr(axis)) / creep_factor
            return self._match_load_shape(EI_values, scalar_input)

        # --- '<module>,<argument>' fallback --------------------------------
        # Only the resolution is guarded, so errors raised inside the
        # resolved function are not reported as an unknown EI_type.
        import importlib

        try:
            module_name, EI_input = EI_type.split(',')
        except ValueError as exc:
            raise ValueError(
                f'Unknown EI_type {EI_type!r}. Expected one of '
                f"{', '.join(self._BUILTIN_EI_TYPES)}, a custom EI, or "
                f"'<module>,<argument>' naming a module that defines a function "
                f'of the same name.') from exc

        try:
            module = importlib.import_module(module_name)
            EI_trial = getattr(module, module_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f'Unknown EI_type {EI_type!r}: could not resolve a function named '
                f'{module_name!r} in a module of that name.') from exc

        # The module must define a function of the same name, called as
        # func(section, axis, arg, betadns, P, M, col).
        return EI_trial(self, axis, EI_input, betadns, P, M, col)

    def interaction_diagram_object(self, axis, num_points=20, factored=False, only_compressive=True):
        # Cache on every input that changes the diagram. A single cached object
        # returned the first diagram built for any later combination, so an
        # x-axis request answered a y-axis one, and factored/only_compressive/
        # num_points changes were ignored.
        key = (axis, num_points, factored, only_compressive)
        if key not in self._CS_id2d:
            from libdenavit import InteractionDiagram2d
            P_CS, M_CS, _ = self.section_interaction_2d(axis, num_points=num_points, factored=factored, only_compressive=only_compressive)
            self._CS_id2d[key] = InteractionDiagram2d(M_CS, P_CS, is_closed=True)
        return self._CS_id2d[key]

    def _invalidate_section_caches(self):
        """
        Drop cached section responses after a change that alters them.

        Only attributes exposed as properties can invalidate automatically.
        fc, fy and conc_cross_section are plain public attributes, so mutating
        them directly cannot be intercepted; call this method explicitly after
        doing so. Converting them to properties is a larger API change and is
        deliberately left out of this commit.
        """
        self._CS_id2d = dict()
        self._Mn = dict()

    def Mn(self, axis):
        if self._Mn.get(axis) is None:
            from libdenavit import InteractionDiagram2d
            P_CS, M_CS, _ = self.section_interaction_2d(axis, 20, factored=False, only_compressive=True)
            CS_id2d = InteractionDiagram2d(M_CS, P_CS, is_closed=True)
            Mn = CS_id2d.find_x_given_y(0, 'pos')
            self._Mn[axis] = Mn

        return self._Mn[axis]

    def phi(self, et):
        f = ACI_phi(self.transverse_reinf_type, et, self.fy / self.Es)
        return f

    def plot_section(self, show=True, **kwargs):
        plt.figure()
        self.conc_cross_section.plot_section(edgecolor='k',facecolor=[0.9,0.9,0.9],**kwargs)
        for i in range(len(self.reinforcement)):
            self.reinforcement[i].plot_section(color='k',**kwargs)
        plt.box(False)
        plt.axis('scaled')
        if show:
            plt.show()

    def aci_strain_compatibility_object(self):
        id_conc = 1
        id_reinf = 2
        fs = self.fiber_section_object(id_conc, id_reinf)
        scACI = AciStrainCompatibility(fs)
        # add concrete boundaries
        x, y, r = self.conc_cross_section.boundary_points
        scACI.add_concrete_boundary(x, y, r)
        # add steel boundary
        for i in self.reinforcement:
            x, y = i.coordinates

            for j in range(len(x)):
                if self._treat_reinforcement_as_point:
                    scACI.add_steel_boundary(x[j], y[j], 0)
                else:
                    raise ValueError("Not implemented yet")

        scACI.max_compressive_strength = -self.pnco
        scACI.add_material(id_conc, 'concrete', self.fc, self.units)
        scACI.add_material(id_reinf, 'steel', self.fy, self.Es)
        return scACI

    def fiber_section_object(self, id_conc, id_reinf, nfx=200, nfy=200):
        fs = FiberSection(nfx, nfy)
        self.conc_cross_section.add_to_fiber_section(fs, id_conc)
        for i in self.reinforcement:
            i.add_to_fiber_section(fs, id_reinf, id_conc)
        return fs

    def section_interaction_2d(self,axis,num_points,factored=False, only_compressive=False):
        scACI = self.aci_strain_compatibility_object()
        scACI.build_data()
        
        if axis == 'x':
            P, M, _, et = scACI.compute_section_interaction_2d(0,num_points,degrees=True)
        elif axis == 'y':
            P, _, M, et = scACI.compute_section_interaction_2d(90,num_points,degrees=True)
        else:
            raise ValueError(f'Unknown axis ({axis})')
            
        if factored:
            ϕ = self.phi(et)
            P = ϕ*P
            M = ϕ*M

        if only_compressive:
            from libdenavit import InteractionDiagram2d
            P = -1 * P
            P_M_id2d = InteractionDiagram2d(M, P, is_closed=True)
            P_et_id2d = InteractionDiagram2d(et, P, is_closed=True)
            M_et_id2d = InteractionDiagram2d(M, et, is_closed=True)
            P = P[:-1]
            M = M[:-1]
            et = et[:-1]
            from libdenavit import find_limit_point_in_list
            ind, x = find_limit_point_in_list(P, 0)
            P = np.insert(P, ind+1, 0)
            M = np.insert(M, ind+1, P_M_id2d.find_x_given_y(0, 'pos'))
            et = np.insert(et, ind+1, P_et_id2d.find_x_given_y(0, 'pos')) # @todo- chcek this

            P = np.insert(P, 0, P_M_id2d.find_y_given_x(0, 'pos'))
            M = np.insert(M, 0, 0)
            et = np.insert(et, 0, M_et_id2d.find_y_given_x(0, 'pos'))# @todo- chcek this

            # Delete extra points
            ind_P_negative = np.where(P < 0)
            P = np.delete(P, ind_P_negative)
            M = np.delete(M, ind_P_negative)
            et = np.delete(et, ind_P_negative)
            ind_M_negative = np.where(M < 0)
            P = np.delete(P, ind_M_negative)
            M = np.delete(M, ind_M_negative)
            et = np.delete(et, ind_M_negative)

            sorted_indices = np.lexsort((-M, P))
            P = np.array(P)[sorted_indices]
            M = np.array(M)[sorted_indices]
            et = np.array(et)[sorted_indices]

        return P, M, et

    def _validate_confinement_inputs(self):
        """
        Validate the transverse reinforcement a confined concrete model needs.

        Run before any confinement arithmetic so that missing or non-physical
        input raises a clear error rather than a TypeError on None, a
        ZeroDivisionError, or a silently inflated strength.
        """
        for name in ('dbt', 's', 'fyt'):
            value = getattr(self, name)
            if value is None:
                raise ValueError(
                    f"Confined concrete requires the transverse reinforcement property "
                    f"'{name}' to be defined on the RC section. Set it via "
                    f"RC(..., {name}=<value>), or use a '_no_confinement' conc_mat_type, "
                    f"which does not require transverse reinforcement.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Transverse reinforcement '{name}' must be a finite positive value "
                    f"(got {value!r})")

        if self.s <= self.dbt:
            raise ValueError(
                f'Hoop spacing s ({self.s}) must be greater than the transverse bar '
                f'diameter dbt ({self.dbt}) so the clear spacing sp = s - dbt is positive.')

        if not np.isfinite(self.fc) or self.fc <= 0:
            raise ValueError(
                f'fc must be a finite positive value for confined concrete (got {self.fc!r})')

    @staticmethod
    def _check_core_reinforcement_ratio(rho_cc):
        """1 - rho_cc divides the confinement effectiveness coefficient."""
        if not np.isfinite(rho_cc) or not 0.0 <= rho_cc < 1.0:
            raise ValueError(
                f'Longitudinal reinforcement ratio of the confined core is {rho_cc!r}; it '
                f'must lie in [0, 1) because 1 - rho_cc appears in a denominator. Check the '
                f'bar area, bar count and core dimensions.')

    @staticmethod
    def _check_confined_core(ke, *confining_stresses):
        """ke must be positive, and each lateral stress is later divided by."""
        if not np.isfinite(ke) or ke <= 0.0:
            raise ValueError(
                f'Confinement effectiveness coefficient ke is {ke!r}; this core geometry '
                f'cannot produce a valid confined concrete model.')
        for fl in confining_stresses:
            if not np.isfinite(fl) or fl <= 0.0:
                raise ValueError(
                    f'Lateral confining stress is {fl!r}; it must be finite and positive. '
                    f'Check the transverse bar area, spacing and yield strength.')

    def confined_concrete_props(self):
        """
        Confined concrete strength and strain.

        Rectangles and circles follow Mander/Chang and Mander; the obround
        applies the circular relations to the core enclosed by each perimeter
        hoop, which is engineering judgement rather than a published model.
        """
        self._validate_confinement_inputs()

        if type(self.conc_cross_section).__name__ == 'Rectangle':
            # dc and bc = core dimensions to centerlines of perimeter hoop in x and y directions
            reinf = self.reinforcement[0]
            for count_name in ('nbx', 'nby'):
                count = getattr(reinf, count_name)
                if count < 2:
                    raise ValueError(
                        f'Confined rectangular sections need at least 2 bars per side; '
                        f'{count_name} is {count}. The clear bar spacing is computed as '
                        f'B/({count_name} - 1) - db.')

            bc = reinf.Bx + reinf.db / 2 + self.dbt / 2
            dc = reinf.By + reinf.db / 2 + self.dbt / 2
            if bc <= 0 or dc <= 0:
                raise ValueError(f'Confined core dimensions must be positive (bc = {bc}, dc = {dc})')

            # Ac = area of core of section enclosed by the center lines of the permiter hoops
            Ac = bc * dc

            # ρcc = ratio of area of longitudinal reinforcement to area of core of section
            ρcc = reinf.Ab * reinf.num_bars / Ac
            self._check_core_reinforcement_ratio(ρcc)

            # wx and wy = clear distance between bars in x and y directions
            wx = reinf.Bx / (reinf.nbx - 1) - reinf.db
            wy = reinf.By / (reinf.nby - 1) - reinf.db
            if wx < 0 or wy < 0:
                raise ValueError(
                    f'Clear distance between longitudinal bars is negative (wx = {wx:.4g}, '
                    f'wy = {wy:.4g}); the bars do not fit in the core as specified.')

            # sp = clear vertical spacing between spiral or hoop bars
            sp = self.s - self.dbt

            # ke = confinement effectiveness coefficient
            sum_w = (2 * (self.reinforcement[0].nbx - 1) * wx ** 2 +
                     2 * (self.reinforcement[0].nby - 1) * wy ** 2) / (6 * bc * dc)
            ke = (1 - sum_w) * (1 - sp / (2 * bc)) * (1 - sp / (2 * dc)) / (1 - ρcc)

            # Asx and Asy = total area of transverse bars running in the x and y directions
            if self.lat_config == 'A':
                Asx = 2 * pi * self.dbt ** 2 / 4
                Asy = 2 * pi * self.dbt ** 2 / 4
            elif self.lat_config == 'B':
                Asx = 4 * pi * self.dbt ** 2 / 4
                Asy = 4 * pi * self.dbt ** 2 / 4
            else:
                raise ValueError(f"Unknown lat_config ({self.lat_config})")

            # flx and fly = lateral confining stress in x and y directions
            flx = ke * Asx / (self.s * dc) * self.fyt
            fly = ke * Asy / (self.s * bc) * self.fyt
            self._check_confined_core(ke, flx, fly)

            # Confinement effect from Chang, G. A., and Mander, J. B. (1994).
            # Seismic Energy Based Fatigue Damage Analysis of Bridge Columns: Part I -
            # Evaluation of Seismic Capacity. National Center for Earthquake Engineering
            # Research, Department of Civil Engineering, State University of New York at
            # Buffalo, Buffalo, New York.
            # Confinement Effect on Strength (Section 3.4.3)
            x_bar = (flx + fly) / (2 * self.fc)
            r = max(flx, fly) / min(flx, fly)
            A_parameter = 6.886 - (0.6069 + 17.275 * r) * exp(-4.989 * r)
            B_parameter = 4.5 / (5 / A_parameter * (0.9849 - 0.6306 * exp(-3.8939 * r)) - 0.1) - 5
            k1 = A_parameter * (0.1 + 0.9 / (1 + B_parameter * x_bar))
            fcc = self.fc * (1 + k1 * x_bar)
            # Confinement Effect on Ductility (Section 3.4.4)
            k2 = 5 * k1
            eps_prime_cc = self.eps_c * (1 + k2 * x_bar)
            return fcc, eps_prime_cc

        if type(self.conc_cross_section).__name__ == 'Circle':
            # ds = core diameter based on center line of perimeter hoop
            ds = 2 * self.reinforcement[0].rc + self.reinforcement[0].db + self.dbt

            # Ac = area of core of section enclosed by the center line of the permiter hoops
            Ac = pi / 4 * ds * ds

            # ρcc = ratio of area of longitudinal reinforcement to area of core of section
            ρcc = self.reinforcement[0].Ab * self.reinforcement[0].num_bars / Ac
            self._check_core_reinforcement_ratio(ρcc)

            # sp = clear vertical spacing between spiral or hoop bars
            sp = self.s - self.dbt

            # ke = confinement effectiveness coefficient (Equation 15)
            ke = (1 - sp / (2 * ds)) / (1 - ρcc)

            # flx and fly = lateral confining stress in x and y directions
            ρs = 4 * self.Abt / (ds * self.s)  # Equation 17
            fl = 0.5 * ke * ρs * self.fyt  # Equation 19
            self._check_confined_core(ke, fl)

            # fcc = confined concrete strength
            fcc = self.fc * (-1.254 + 2.254 * sqrt(1 + 7.94 * fl / self.fc) - 2 * fl / self.fc)

            # Confinement Effect on Ductility (Section 3.4.4 of Chang and Mander 1994)
            k1 = (fcc - self.fc) / fl
            k2 = 5 * k1
            x_bar = (fl + fl) / (2 * self.fc)
            eps_prime_cc = self.eps_c * (1 + k2 * x_bar)
            return fcc, eps_prime_cc

        if type(self.conc_cross_section).__name__ == 'Obround':
            # ds = core diameter based on center line of perimeter hoop
            ds = self.reinforcement[0].D + self.reinforcement[0].db + self.dbt

            # Ac = area of core of section enclosed by the center line of the perimeter hoops
            Ac = pi / 4 * ds * ds

            # ρcc = ratio of area of longitudinal reinforcement to area of core of section
            ρcc = self.reinforcement[0].Ab * self.reinforcement[0].num_bars / 2 / Ac
            self._check_core_reinforcement_ratio(ρcc)

            # sp = clear vertical spacing between spiral or hoop bars
            sp = self.s - self.dbt

            # ke = confinement effectiveness coefficient (Equation 15)
            ke = (1 - sp / (2 * ds)) / (1 - ρcc)

            # flx and fly = lateral confining stress in x and y directions
            ρs = 4 * self.Abt / (ds * self.s)  # Equation 17
            fl = 0.5 * ke * ρs * self.fyt  # Equation 19
            self._check_confined_core(ke, fl)

            # fcc = confined concrete strength
            fcc = self.fc * (-1.254 + 2.254 * sqrt(1 + 7.94 * fl / self.fc) - 2 * fl / self.fc)

            # Confinement Effect on Ductility (Section 3.4.4 of Chang and Mander 1994)
            k1 = (fcc - self.fc) / fl
            k2 = 5 * k1
            x_bar = (fl + fl) / (2 * self.fc)
            eps_prime_cc = self.eps_c * (1 + k2 * x_bar)
            return fcc, eps_prime_cc

        raise ValueError(
            f'No confined concrete model implemented for cross section type '
            f'{type(self.conc_cross_section).__name__}')

    def build_ops_fiber_section(self, section_id, start_material_id, steel_mat_type, conc_mat_type, nfy, nfx, GJ=1.0e6,
                                axis=None, creep=False, creep_props_dict=None, shrinkage_props_dict=None):
        """ Builds the fiber section object

        Parameters
        ----------
        section_id : int
            The id of the section
        start_material_id : int
            The id of the first uniaxial material to be defined (others will be defined sequentially)
        steel_mat_type : str
            The type of the steel material
        conc_mat_type : str
            The type of the concrete material
        nfy : int
            The minimum number of fibers in the y direction
        nfx : int
            The minimum number of fibers in the x direction
        GJ : float (default 1.0e6)
            The torsional rigidity of the cross section
        axis : str (default None)
            Optional argument for defining a fiber section for 2-dimensional analysis
              - If "None", then a 3-dimensional fiber section will be defined
              - If "x", then a 2-dimensional fiber section will be defined based on 
                bending about the x-axis (note that the value nfx will be ignored) 
              - If "y", then a 2-dimensional fiber section will be defined based on 
                bending about the y-axis (note that the value nfy will be ignored)
        creep: bool (default False)
            Optional argument to include creep parameters in concrete definition
        """

        for _name, _value in (('nfy', nfy), ('nfx', nfx)):
            if not isinstance(_value, (int, np.integer)) or isinstance(_value, bool) or _value < 1:
                raise ValueError(f'{_name} must be a positive integer (got {_value!r})')

        # region Define Material IDs
        steel_material_id = start_material_id
        concrete_material_id = start_material_id + 1  # Used if no confinement
        cover_concrete_material_id = start_material_id + 1  # Used if confinement
        core_concrete_material_id = start_material_id + 2  # Used if confinement
        concrete_creep_material_id = start_material_id + 3
        cover_concrete_creep_material_id = start_material_id + 3
        core_concrete_creep_material_id = start_material_id + 4
        # endregion

        # region Define OpenSees Section
        ops.section('Fiber', section_id, '-GJ', GJ)
        # endregion

        # region Define Steel Material
        if steel_mat_type == "ElasticPP":
            ops.uniaxialMaterial("ElasticPP", steel_material_id, self.Es, self.fy / self.Es)

        elif steel_mat_type == "Hardening":
            ops.uniaxialMaterial("Hardening", steel_material_id, self.Es, self.fy, 0.001*self.Es, 0)

        elif steel_mat_type == "ReinforcingSteel":
            ops.uniaxialMaterial("ReinforcingSteel", steel_material_id, self.fy, self.fy * 1.5, self.Es, self.Es / 2,
                                 0.002, 0.008)

        elif steel_mat_type == "Elastic":
            ops.uniaxialMaterial("Elastic", steel_material_id, self.Es)

        else:
            raise ValueError(f"Steel material {steel_mat_type} not supported")
        # endregion

        # region Define Concrete Material
        # Reinforcement pattern required by each concrete cross-section type.
        supported_reinf_for_section = {
            'Rectangle': 'ReinfRect',
            'Circle': 'ReinfCirc',
            'Obround': 'ReinfIntersectingLoops',
        }

        def validate_section_reinf(self):
            section_name = type(self.conc_cross_section).__name__
            expected = supported_reinf_for_section.get(section_name)
            if expected is not None:
                actual = type(self.reinforcement[0]).__name__
                if actual != expected:
                    # Report the offending pattern type, not the container type:
                    # type(self.reinforcement).__name__ is always 'list'.
                    raise ValueError(
                        f"Reinforcement type {actual} not supported for section type "
                        f"{section_name} (expected {expected})")

            if self.reinforcement[0].xc != 0 or self.reinforcement[0].yc != 0:
                raise ValueError(f"Reinforcing pattern must be centered")

        def validate_confinement_reinf(self):
            # Delegates so the transverse reinforcement, spacing and core
            # geometry are checked identically wherever confinement is used.
            try:
                self._validate_confinement_inputs()
            except ValueError as exc:
                raise ValueError(
                    f"conc_mat_type='{conc_mat_type}' models confined concrete: {exc}") from exc

        confinement = False

        if conc_mat_type == "Concrete04":
            ''' 
                Defined based on Mander, J. B., Priestley, M. J. N., and Park, R. (1988). 
                “Theoretical Stress-Strain Model for Confined Concrete.” Journal of Structural 
                Engineering, ASCE, 114(8), 1804―1826.
            '''

            validate_section_reinf(self)
            validate_confinement_reinf(self)

            fcc, eps_prime_cc = self.confined_concrete_props()

            ops.uniaxialMaterial("Concrete04", cover_concrete_material_id, -self.fc, -self.eps_c, -1.0, self.Ec)
            ops.uniaxialMaterial("Concrete04", core_concrete_material_id, -fcc, -eps_prime_cc, -1.0, self.Ec)
            confinement = True

        elif conc_mat_type == "Concrete04_no_confinement":
            validate_section_reinf(self)
            ops.uniaxialMaterial("Concrete04", concrete_material_id, -self.fc, -self.eps_c, -1.0, self.Ec)

        elif conc_mat_type == "Concrete02":
            validate_section_reinf(self)
            validate_confinement_reinf(self)

            fcc, eps_prime_cc = self.confined_concrete_props()

            ops.uniaxialMaterial('Concrete02', cover_concrete_material_id, -self.fc, -self.eps_c, -self.fc/1.5, -0.01)
            ops.uniaxialMaterial('Concrete02', core_concrete_material_id, -fcc, -eps_prime_cc, -fcc/1.5, -0.01)
            confinement = True

        elif conc_mat_type == "Concrete02_no_confinement":
            validate_section_reinf(self)
            ops.uniaxialMaterial('Concrete02', concrete_material_id, -self.fc, -self.eps_c, -self.fc/1.5, -0.01)
        
        elif conc_mat_type == "Concrete01_no_confinement":
            validate_section_reinf(self)
            ops.uniaxialMaterial("Concrete01", concrete_material_id, -self.fc, -2 * self.fc / self.Ec, 0, 0.006)

        elif conc_mat_type == "ENT":
            ops.uniaxialMaterial('ENT', concrete_material_id, self.Ec)

        elif conc_mat_type == "Elastic":
            ops.uniaxialMaterial('Elastic', concrete_material_id, self.Ec)

        else:
            raise ValueError(f"Concrete material {conc_mat_type} not supported")
        # endregion

        # region Define Creep Material
        if creep:
            creepdata = self.get_creep_props_for_uniaxial_material(**(creep_props_dict or {}))
            shrinkagedata = self.get_shrinkage_props_for_uniaxial_material(**(shrinkage_props_dict or {}))

            if self._default_age_warn:
                if self.tD == 5:
                    warnings.warn("Default value of tD (5) used for creep and shrinkage material")
                if self.Tcr == 28:
                    warnings.warn("Default value of Tcr (28) used for creep and shrinkage material")
                if self.tcast == 0:
                    warnings.warn("Default value of tcast (0) used for creep and shrinkage material")

            if confinement:
                creep_pairs = ((cover_concrete_creep_material_id, cover_concrete_material_id),
                               (core_concrete_creep_material_id, core_concrete_material_id))
            else:
                creep_pairs = ((concrete_creep_material_id, concrete_material_id),)

            for creep_material_id, base_material_id in creep_pairs:
                ops.uniaxialMaterial('CreepShrinkageACI209', creep_material_id, base_material_id,
                                     self.tD, shrinkagedata['eps_sh_u'], shrinkagedata['psish'], self.Tcr,
                                     creepdata['phi_u'], creepdata['psicr1'], creepdata['psicr2'], self.tcast)

            if confinement:
                cover_concrete_material_id = cover_concrete_creep_material_id
                core_concrete_material_id = core_concrete_creep_material_id
            else:
                concrete_material_id = concrete_creep_material_id
        # endregion

        # region Define fibers and patches
        if type(self.conc_cross_section).__name__ == 'Rectangle':
            if (axis is None) or (axis == '3d'):
                for i in self.reinforcement:
                    for index, value in enumerate(i.coordinates[0]):
                        ops.fiber(i.coordinates[1][index], value, i.Ab, steel_material_id)
                        if confinement:
                            negative_area_material_id = core_concrete_material_id
                        else:
                            negative_area_material_id = concrete_material_id
                        ops.fiber(i.coordinates[1][index], value, -i.Ab, negative_area_material_id)

                H = self.conc_cross_section.H
                B = self.conc_cross_section.B
                cdb = (H - self.reinforcement[0].By) / 2 - self.reinforcement[0].db / 2
                if self.dbt is not None:
                    cdb -= self.dbt / 2

                if confinement:
                    nfy_cover = ceil(cdb * nfy / H)
                    nfx_cover = ceil(cdb * nfx / B)
                    nfy_core = ceil((H - 2 * cdb) * nfy / H)
                    nfx_core = ceil((B - 2 * cdb) * nfx / B)

                    ops.patch('rect', cover_concrete_material_id, nfy_cover, nfx,
                              -H / 2, -B / 2, -H / 2 + cdb, B / 2)

                    ops.patch('rect', cover_concrete_material_id, nfy_core, nfx_cover,
                              -H / 2 + cdb, -B / 2, H / 2 - cdb, -B / 2 + cdb)

                    ops.patch('rect', cover_concrete_material_id, nfy_cover, nfx,
                              H / 2 - cdb, -B / 2, H / 2, B / 2)

                    ops.patch('rect', cover_concrete_material_id, nfy_core, nfx_cover,
                              -H / 2 + cdb, B / 2 - cdb, H / 2 - cdb, B / 2)

                    ops.patch('rect', core_concrete_material_id, nfy_core, nfx_core,
                              -H / 2 + cdb, -B / 2 + cdb, H / 2 - cdb, B / 2 - cdb)
                else:
                    ops.patch('rect', concrete_material_id, nfy, nfx, -H / 2, -B / 2, H / 2, B / 2)

            elif axis in ['x', 'y']:
                # H is the section depth in the direction of bending, B the
                # width perpendicular to it, and nfd the fiber count through
                # the depth (nfx is ignored for x bending, nfy for y bending).
                # reinf_depth is the reinforcement extent measured through the
                # same depth direction: By for x bending, Bx for y bending.
                if axis == 'x':
                    H = self.conc_cross_section.H
                    B = self.conc_cross_section.B
                    nfd = nfy
                    reinf_depth = self.reinforcement[0].By
                else:
                    H = self.conc_cross_section.B
                    B = self.conc_cross_section.H
                    nfd = nfx
                    reinf_depth = self.reinforcement[0].Bx

                for i in self.reinforcement:
                    rebar_coords = i.coordinates[1] if axis == 'x' else i.coordinates[0]

                    for index, value in enumerate(rebar_coords):
                        ops.fiber(float(value), 0, i.Ab, steel_material_id)
                        if confinement:
                            negative_area_material_id = core_concrete_material_id
                        else:
                            negative_area_material_id = concrete_material_id
                        ops.fiber(float(value), 0, -i.Ab, negative_area_material_id)

                cdb = (H - reinf_depth) / 2 - self.reinforcement[0].db / 2
                if self.dbt is not None:
                    cdb = cdb - self.dbt / 2
                def strip_layer(material_id, n_fibers, y_start, y_end):
                    n_fibers = max(1, int(n_fibers))
                    strip_height = (y_end - y_start) / n_fibers
                    half = strip_height / 2
                    ops.layer('straight', material_id, n_fibers, strip_height * B,
                              y_start + half, 0, y_end - half, 0)

                if confinement:
                    nfd_cover = max(1, ceil(cdb * nfd / H))
                    nfd_core = max(1, ceil((H - 2 * cdb) * nfd / H))

                    strip_layer(core_concrete_material_id, nfd_core, -H / 2 + cdb, H / 2 - cdb)
                    strip_layer(cover_concrete_material_id, nfd_cover, -H / 2, -H / 2 + cdb)
                    strip_layer(cover_concrete_material_id, nfd_cover, H / 2 - cdb, H / 2)
                else:
                    strip_layer(concrete_material_id, nfd, -H / 2, H / 2)

            else:
                raise ValueError(f'Unknown option for axis: {axis}')

        elif type(self.conc_cross_section).__name__ == 'Circle':
            if (axis is None) or (axis == '3d'):
                for i in self.reinforcement:
                    for index, value in enumerate(i.coordinates[0]):
                        ops.fiber(i.coordinates[1][index], value, i.Ab, steel_material_id)
                        if confinement:
                            negative_area_material_id = core_concrete_material_id
                        else:
                            negative_area_material_id = concrete_material_id
                        ops.fiber(i.coordinates[1][index], value, -i.Ab, negative_area_material_id)

                d = self.conc_cross_section.diameter
                max_fiber_size = d / max(nfx, nfy)
                if confinement:
                    ds = 2*self.reinforcement[0].rc + self.reinforcement[0].db + self.dbt
                    # Core Concrete
                    nfr = ceil(0.125*ds/max_fiber_size)              
                    nfc = ceil(0.25*ds*pi/max_fiber_size)
                    ops.patch('circ', core_concrete_material_id, nfc, nfr, 0, 0, 0,        0.125*ds, 0, 360)
                    nfc = ceil(0.5*ds*pi/max_fiber_size)
                    ops.patch('circ', core_concrete_material_id, nfc, nfr, 0, 0, 0.125*ds, 0.250*ds, 0, 360)
                    nfc = ceil(0.75*ds*pi/max_fiber_size)
                    ops.patch('circ', core_concrete_material_id, nfc, nfr, 0, 0, 0.250*ds, 0.375*ds, 0, 360)
                    nfc = ceil(ds*pi/max_fiber_size)
                    ops.patch('circ', core_concrete_material_id, nfc, nfr, 0, 0, 0.375*ds, 0.500*ds, 0, 360)
                    # Cover Concrete
                    nfr = ceil(0.5*(d-ds)/max_fiber_size)
                    nfc = ceil(d*pi/max_fiber_size)
                    ops.patch('circ', cover_concrete_material_id, nfc, nfr, 0, 0, 0.5*ds, 0.5*d, 0, 360)
                else:
                    nfr = ceil(0.125*d/max_fiber_size)              
                    nfc = ceil(0.25*d*pi/max_fiber_size)
                    ops.patch('circ', concrete_material_id, nfc, nfr, 0, 0, 0,       0.125*d, 0, 360)
                    nfc = ceil(0.5*d*pi/max_fiber_size)
                    ops.patch('circ', concrete_material_id, nfc, nfr, 0, 0, 0.125*d, 0.250*d, 0, 360)
                    nfc = ceil(0.75*d*pi/max_fiber_size)
                    ops.patch('circ', concrete_material_id, nfc, nfr, 0, 0, 0.250*d, 0.375*d, 0, 360)
                    nfc = ceil(d*pi/max_fiber_size)
                    ops.patch('circ', concrete_material_id, nfc, nfr, 0, 0, 0.375*d, 0.500*d, 0, 360)
            elif axis in ['x', 'y']:
                for i in self.reinforcement:
                    if axis == 'x':
                        rebar_coords = i.coordinates[1]
                    else:
                        rebar_coords = i.coordinates[0]

                    for index, value in enumerate(rebar_coords):
                        ops.fiber(value, 0, i.Ab, steel_material_id)
                        if confinement:
                            negative_area_material_id = core_concrete_material_id
                        else:
                            negative_area_material_id = concrete_material_id
                        ops.fiber(value, 0, -i.Ab, negative_area_material_id)

                d = self.conc_cross_section.diameter
                # nfx is ignored for x bending, nfy for y bending
                max_fiber_size = d / (nfy if axis == 'x' else nfx)
                if confinement:
                    ds = 2*self.reinforcement[0].rc + self.reinforcement[0].db + self.dbt
                    nf = ceil(0.5*ds/max_fiber_size)
                    circ_patch_2d(core_concrete_material_id, nf, ds)
                    nf = ceil(0.5*d/max_fiber_size)
                    circ_patch_2d(cover_concrete_material_id, nf, d, Di=ds)
                else:
                    nf = ceil(0.5*d/max_fiber_size)
                    circ_patch_2d( concrete_material_id, nf, d)
            else:
                raise ValueError(f'Unknown option for axis: {axis}')

        elif type(self.conc_cross_section).__name__ == 'Obround':
            if (axis is None) or (axis == '3d'):
                raise ValueError(f'3D option not supported for obround cross-sections yet')

            elif axis == 'x' or axis == 'y':
                if confinement:
                    negative_area_material_id = core_concrete_material_id
                else:
                    negative_area_material_id = concrete_material_id

                for i in self.reinforcement:
                    if axis == 'x':
                        for index, value in enumerate(i.coordinates[1]):
                            ops.fiber(value, 0, i.Ab, steel_material_id)
                            ops.fiber(value, 0, -i.Ab, negative_area_material_id)
                    elif axis == 'y':
                        for index, value in enumerate(i.coordinates[0]):
                            ops.fiber(value, 0, i.Ab, steel_material_id)
                            ops.fiber(value, 0, -i.Ab, negative_area_material_id)

                # nfx is ignored for x bending, nfy for y bending
                nf = nfy if axis == 'x' else nfx

                if confinement:
                    ds = self.reinforcement[0].D + self.reinforcement[0].db + self.dbt
                    obround_patch_2d_confined(cover_concrete_material_id, core_concrete_material_id, nf,
                                              self.conc_cross_section.D, self.conc_cross_section.a, ds,
                                              axis=axis)
                else:
                    obround_patch_2d(concrete_material_id, nf,
                                     self.conc_cross_section.D, self.conc_cross_section.a,
                                     axis=axis)

            else:
                raise ValueError(f'Unknown option for axis: {axis}')

        else:
            raise ValueError(f"Concrete cross section {self.conc_cross_section.section_type} not supported")
        # endregion


def run_example():
    from libdenavit.section import Rectangle, ReinfRect

    # Define concrete cross section
    H = 40
    B = 20
    conc_cross_section = Rectangle(H, B)

    # Define reinforcement
    rhosr = 0.06
    nbB = 2
    nbH = 3
    cover = 0.15 * H
    Ab = H * B * rhosr / (2 * nbB + 2 * nbH - 4)
    reinforcement = ReinfRect(B - 2 * cover, H - 2 * cover, nbB, nbH, Ab)

    # Define materials
    fc = 4
    fy = 60
    units = 'US'

    # Define RC object
    section = RC(conc_cross_section, reinforcement, fc, fy, units)
    
    # Plot Section
    section.plot_section(show=False)

    # Plot Interaction Diagram
    angle = 0
    num_points = 40
    P, M, et = section.section_interaction_2d("x", num_points)
    ϕ = section.phi(et)
   
    plt.figure()
    plt.plot(M, -P, '-o', label="$M_x$ (nominal)")
    plt.plot(ϕ*M, -ϕ*P, '-s', label="$M_x$ (design)")
    plt.xlabel('Bending Moment (kip-in.)')
    plt.ylabel('Axial Compression (kips)')
    plt.legend()
    plt.show()


def run_example_2():
    D = 36
    a = 18
    Dc = D - 6
    nb = 20
    Ab = 1

    from libdenavit.section import Obround, ReinfIntersectingLoops
    conc_cross_section = Obround(D, a)

    # Define reinforcement
    reinforcement = ReinfIntersectingLoops(Dc, a, nb, Ab)

    # Define materials
    fc = 4
    fy = 60
    units = 'US'
    axis = 'y'
    # Define RC object
    section = RC(conc_cross_section, reinforcement, fc, fy, units)

    # Plot Section
    section.plot_section(show=False)

    # Plot Interaction Diagram
    angle = 0
    num_points = 40
    P, M, et = section.section_interaction_2d('x', num_points)
    ϕ = section.phi(et)

    plt.figure()
    plt.plot(M, -P, '-o', label="$M_x$ (nominal)")
    plt.plot(ϕ * M, -ϕ * P, '-s', label="$M_x$ (design)")
    plt.xlabel('Bending Moment (kip-in.)')
    plt.ylabel('Axial Compression (kips)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # run_example()
    run_example_2()
