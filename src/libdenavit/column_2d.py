from abc import ABC, abstractmethod
from libdenavit.OpenSees import AnalysisResults
from libdenavit import interpolate_list, find_limit_point_in_list
from libdenavit.analysis_helpers import (
    CONCRETE_COMPRESSION_STRAIN_LIMIT, DEFORMATION_LIMIT, EIGENVALUE_LIMIT,
    LEGACY_LIMIT_MESSAGES, STEEL_COMPRESSION_STRAIN_LIMIT, STEEL_TENSILE_STRAIN_LIMIT,
)
import warnings

class Column2d(ABC):
    """
    A base class for 2D column models.
    """
    # Keyword arguments consumed by __init__. Subclasses extend this with their own,
    # which lets a misspelled or unsupported argument be reported instead of silently
    # ignored.
    _INIT_KWARGS = frozenset({
        'axis', 'dxo', 'include_initial_geometric_imperfections',
        'ops_n_elem', 'ops_element_type',
        'ops_geom_transf_type', 'ops_integration_points',
    })

    def __init__(self, section, length, **kwargs):
        """
        Initializes common properties for a column.

        Args:
            section: The cross-section object.
            length (float): The length of the column.
            **kwargs: Additional keyword arguments for customization.
        """
        self.section = section
        self.length = length

        # Common analysis and geometry parameters with default values
        self.axis = kwargs.get('axis', None)
        self.ops_n_elem = kwargs.get('ops_n_elem', 8)
        self.ops_element_type = kwargs.get('ops_element_type', 'mixedBeamColumn')
        self.ops_geom_transf_type = kwargs.get('ops_geom_transf_type', 'Corotational')
        self.ops_integration_points = kwargs.get('ops_integration_points', 3)
        self.dxo = kwargs.get('dxo', 0.0)
        self.include_initial_geometric_imperfections = kwargs.get('include_initial_geometric_imperfections', True)
        
        unknown = set(kwargs) - self._INIT_KWARGS
        if unknown:
            warnings.warn(f'{type(self).__name__} ignoring unrecognized keyword '
                          f'argument(s): {sorted(unknown)}')


    def _init_results(self, attrs: list) -> AnalysisResults:
        """Create an AnalysisResults with the given attribute names as empty lists."""
        results = AnalysisResults()
        for name in attrs:
            setattr(results, name, [])
        return results
    
    def _get_default_deformation_limit(self):
        """Get default deformation limit. Override in subclasses."""
        return 0.1 * self.length
    
    def _extract_analysis_config(self, **kwargs):
        """Extract and validate analysis configuration parameters."""
        config = {
            'section_id': kwargs.get('section_id', 1),
            'section_args': kwargs.get('section_args', []),
            'section_kwargs': kwargs.get('section_kwargs', {}),
            'e': kwargs.get('e', 1.0),
            'P': kwargs.get('P', 0),
            'num_steps_vertical': kwargs.get('num_steps_vertical', 1000),
            'disp_incr_factor': kwargs.get('disp_incr_factor', 1e-5),
            'axial_control': kwargs.get('axial_control', 'load'),
            'eigenvalue_limit': kwargs.get('eigenvalue_limit', 0),
            'deformation_limit': kwargs.get('deformation_limit', 'default'),
            'concrete_strain_limit': kwargs.get('concrete_strain_limit', -0.01),
            'steel_strain_limit': kwargs.get('steel_strain_limit', 0.05),
            'percent_load_drop_limit': kwargs.get('percent_load_drop_limit', 0.05),
            'try_smaller_steps': kwargs.get('try_smaller_steps', True),
            'creep_props_dict': kwargs.get('creep_props_dict', dict()),
            'shrinkage_props_dict': kwargs.get('shrinkage_props_dict', dict()),
        }
        
        # Set default deformation limit using subclass override
        if config['deformation_limit'] == 'default':
            config['deformation_limit'] = self._get_default_deformation_limit()

        if config['axial_control'] not in ('load', 'displacement'):
            raise ValueError(f"axial_control must be 'load' or 'displacement', "
                             f"got {config['axial_control']!r}")

        return config
    
    def _initialize_results(self):
        """Initialize analysis results object with required attributes. Override in subclasses."""
        # Base attributes that most column types will need
        attrs = [
            'applied_axial_load', 'maximum_abs_moment', 'maximum_abs_disp', 'lowest_eigenvalue',
            'maximum_concrete_compression_strain', 'maximum_steel_strain', 'time_in_longterm_analysis', 'deformation_in_longterm_analysis',
            'maximum_compression_strain', 'maximum_tensile_strain', 'curvature'
        ]
        return self._init_results(attrs)
    
    def _set_limit_point_values(self, results, ind, x):
        """Set limit point values. Override in subclasses for additional values."""
        results.applied_axial_load_at_limit_point = interpolate_list(results.applied_axial_load, ind, x)
        results.maximum_abs_moment_at_limit_point = interpolate_list(results.maximum_abs_moment, ind, x)
        results.maximum_abs_disp_at_limit_point = interpolate_list(results.maximum_abs_disp, ind, x)
    
    def _peak_response(self, results, analysis_type=None):
        """Series and target for messages that name no quantity: the peak applied load.

        Override where a different load drives the analysis.
        """
        return results.applied_axial_load, max(results.applied_axial_load)

    def _limit_point_sources(self, results, config):
        """Map each limit message to the (series, target) whose crossing sets the limit point."""
        is_rc = type(self.section).__name__ == 'RC'
        return {
            EIGENVALUE_LIMIT:
                (results.lowest_eigenvalue, config['eigenvalue_limit']),
            DEFORMATION_LIMIT:
                (results.maximum_abs_disp, config['deformation_limit']),
            CONCRETE_COMPRESSION_STRAIN_LIMIT:
                (results.maximum_concrete_compression_strain, config['concrete_strain_limit']),
            # Compression strains are negative, so the limit is crossed going down.
            STEEL_COMPRESSION_STRAIN_LIMIT:
                (results.maximum_compression_strain, -config['steel_strain_limit']),
            # RC records the bar strain, I_shape the extreme tensile fiber strain.
            STEEL_TENSILE_STRAIN_LIMIT:
                (results.maximum_steel_strain if is_rc else results.maximum_tensile_strain,
                 config['steel_strain_limit']),
        }

    def _find_limit_point(self, results, config, analysis_type=None):
        """Locate the limit point and store its values on results.

        A message naming a quantity is resolved through _limit_point_sources, so the
        limit point is interpolated at the exact crossing. Everything else, including
        analysis failures and load drops, falls back to the peak applied load.
        """
        if not results.exit_message:
            results.exit_message = 'Analysis Ended Without Explicit Limit'

        if 'analysis failed' in results.exit_message.lower():
            warnings.warn(f'Analysis failed: {results.exit_message}')

        message = LEGACY_LIMIT_MESSAGES.get(results.exit_message, results.exit_message)
        source = self._limit_point_sources(results, config).get(message)
        if source is None:
            source = self._peak_response(results, analysis_type)

        ind, x = find_limit_point_in_list(*source)
        self._set_limit_point_values(results, ind, x)
    
    def run_ops_analysis(self, analysis_type, **kwargs):
        """Template method for running OpenSees analysis."""
        config = self._extract_analysis_config(**kwargs)

        if (analysis_type.lower() == 'proportional_limit_point' and config['e'] == 0
                and not self.dxo and not getattr(self, 'Dxo', 0)):
            warnings.warn(
                'Running a proportional_limit_point analysis with e=0 (axial-only loading) and '
                'no initial geometric imperfection (dxo=0, and Dxo=0 if applicable): the model is '
                'perfectly symmetric, so there is nothing to trigger a second-order (buckling) '
                'response and the analysis may fail to converge. If you intend to capture buckling '
                'behavior, pass a nonzero dxo (e.g. dxo=length/1000) when constructing the column -- '
                'or, for a SwayColumn2d, a nonzero Dxo, since dxo alone is zero at the sway-controlled '
                'top node.'
            )

        self.build_ops_model(config['section_id'], config['section_args'], config['section_kwargs'],
                            creep_props_dict=config['creep_props_dict'],
                            shrinkage_props_dict=config['shrinkage_props_dict'])
        
        results = self._initialize_results()
        
        # Delegate to specific analysis implementation
        if analysis_type.lower() == 'proportional_limit_point':
            results = self._run_proportional_analysis(config, results)
        elif analysis_type.lower() == 'nonproportional_limit_point':
            results = self._run_nonproportional_analysis(config, results)
        else:
            raise ValueError(f'Analysis type {analysis_type} not implemented')
        
        self._find_limit_point(results, config, analysis_type)
        return results

    @abstractmethod
    def _run_proportional_analysis(self, config, results):
        """Run proportional analysis. Must be implemented by subclasses."""
        pass

    @abstractmethod  
    def _run_nonproportional_analysis(self, config, results):
        """Run nonproportional analysis. Must be implemented by subclasses."""
        pass
    