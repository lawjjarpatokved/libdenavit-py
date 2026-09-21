from libdenavit import opensees as ops


EIGENVALUE_LIMIT = 'Eigenvalue Limit Reached'
DEFORMATION_LIMIT = 'Deformation Limit Reached'
CONCRETE_COMPRESSION_STRAIN_LIMIT = 'Concrete Compression Strain Limit Reached'
STEEL_COMPRESSION_STRAIN_LIMIT = 'Steel Compression Strain Limit Reached'
STEEL_TENSILE_STRAIN_LIMIT = 'Steel Tensile Strain Limit Reached'

# Earlier spellings, kept so stored results and outside code still resolve.
LEGACY_LIMIT_MESSAGES = {
    'Extreme Compressive Concrete Fiber Strain Limit Reached': CONCRETE_COMPRESSION_STRAIN_LIMIT,
    'Extreme Steel Fiber Strain Limit Reached': STEEL_TENSILE_STRAIN_LIMIT,
}


def try_analysis_options():
    """
    Tries different analysis algorithms and tolerances in OpenSees.

    This function attempts to converge the analysis using a sequence of
    different algorithms ('ModifiedNewton', 'KrylovNewton') and tolerances.

    Returns:
        int: The status of the analysis (0 if successful, -1 if failed).
    """
    options = [('ModifiedNewton', 1e-3),
               ('KrylovNewton', 1e-3),
               ('KrylovNewton', 1e-2)]

    for algorithm, tolerance in options:
        ops.algorithm(algorithm)
        ops.test('NormUnbalance', tolerance, 10)
        ok = ops.analyze(1)
        if ok == 0:
            break
    return ok


def ops_get_section_strains(column):
    """
    Get section strains for both RC and I_shape sections
    Returns: [maximum_compression_strain, maximum_tensile_strain, curvatureX, curvatureY]
    """
    if type(column.section).__name__ == "RC":
        # Original RC implementation
        maximum_concrete_compression_strain = []
        maximum_tensile_steel_strain = []
        max_kx = 0.0
        max_ky = 0.0
        for i in range(column.ops_n_elem):
            for j in range(column.ops_integration_points):
                axial_strain, curvatureX, curvatureY = 0, 0, 0
                if column.axis == 'x':
                    axial_strain, curvatureX = ops.eleResponse(i,  # element tag
                                                              'section', j+1,  # select integration point
                                                              'deformation')  # response type
                elif column.axis == 'y':
                    axial_strain, curvatureY = ops.eleResponse(i,  # element tag
                                                              'section', j+1,  # select integration point
                                                              'deformation')  # response type
                else:
                    raise ValueError("The axis is not supported.")
                
                maximum_concrete_compression_strain.append(column.section.maximum_concrete_compression_strain(
                                                          axial_strain, curvatureX=curvatureX, curvatureY=curvatureY))
                maximum_tensile_steel_strain.append(column.section.maximum_tensile_steel_strain(
                                                   axial_strain, curvatureX=curvatureX, curvatureY=curvatureY))
                max_kx = max(max_kx, abs(curvatureX))
                max_ky = max(max_ky, abs(curvatureY))

        return min(maximum_concrete_compression_strain), max(maximum_tensile_steel_strain), max_kx, max_ky

    elif type(column.section).__name__ == "I_shape":
        # I_shape implementation using section methods
        maximum_compression_strains = []
        maximum_tensile_strains = []
        max_kx = 0.0
        max_ky = 0.0

        for i in range(column.ops_n_elem):
            for j in range(column.ops_integration_points):
                axial_strain, curvatureX, curvatureY = 0, 0, 0
                if column.axis == 'x':
                    axial_strain, curvatureX = ops.eleResponse(i,  # element tag
                                                              'section', j+1,  # select integration point
                                                              'deformation')  # response type
                elif column.axis == 'y':
                    axial_strain, curvatureY = ops.eleResponse(i,  # element tag
                                                              'section', j+1,  # select integration point
                                                              'deformation')  # response type
                else:
                    raise ValueError("The axis is not supported.")
                
                # Use I_shape methods
                maximum_compression_strains.append(column.section.maximum_compression_strain(
                                                  axial_strain, curvatureX=curvatureX, curvatureY=curvatureY))
                maximum_tensile_strains.append(column.section.maximum_tensile_strain(
                                              axial_strain, curvatureX=curvatureX, curvatureY=curvatureY))
                max_kx = max(max_kx, abs(curvatureX))
                max_ky = max(max_ky, abs(curvatureY))

        return min(maximum_compression_strains), max(maximum_tensile_strains), max_kx, max_ky


def ops_get_maximum_abs_moment(column) -> float:
    """
    Gets the maximum absolute moment from the column elements in OpenSees.

    Args:
        column: The column object with analysis properties.

    Returns:
        float: The maximum absolute moment.
    """
    moment = [abs(ops.eleForce(0, 3))]  # Moment at the start of the first element
    for i in range(column.ops_n_elem):
        moment.append(abs(ops.eleForce(i, 6))) # Moment at the end of each element
    return max(moment)


def ops_get_maximum_abs_disp(column) -> float:
    """
    Gets the maximum absolute lateral displacement from the nodes in OpenSees.

    Args:
        column: The column object with analysis properties.

    Returns:
        float: The maximum absolute lateral displacement.
    """
    return max(abs(ops.nodeDisp(i, 1)) for i in range(column.ops_n_elem + 1))


def check_analysis_limits(results, **limits):
    """
    Checks if the analysis has reached a predefined limit.

    Args:
        results: The AnalysisResults object containing the current state.
        limits (dict): A dictionary of limit values.

    Returns:
        str or None: An exit message string if a limit is reached, otherwise None.
    """
    # Unpack limits from the dictionary
    eigenvalue_limit = limits.get('eigenvalue_limit')
    deformation_limit = limits.get('deformation_limit')
    concrete_strain_limit = limits.get('concrete_strain_limit')
    steel_strain_limit = limits.get('steel_strain_limit')
    
    # Optional; defaults to RC so existing callers keep working.
    section_type = limits.get('section_type', 'RC')

    if eigenvalue_limit is not None and results.lowest_eigenvalue[-1] < eigenvalue_limit:
        return EIGENVALUE_LIMIT

    if deformation_limit is not None and results.maximum_abs_disp[-1] > deformation_limit:
            return DEFORMATION_LIMIT
    
    if section_type == "RC":
        # For RC: Check concrete compression and steel tension
        if concrete_strain_limit is not None and results.maximum_concrete_compression_strain[-1] < concrete_strain_limit:
            return CONCRETE_COMPRESSION_STRAIN_LIMIT
        
        if steel_strain_limit is not None and results.maximum_steel_strain[-1] > steel_strain_limit:
            return STEEL_TENSILE_STRAIN_LIMIT
        
    elif section_type == "I_shape":
        # compression strains are negative; exceed limit if magnitude > steel_strain_limit
        if steel_strain_limit is not None and results.maximum_compression_strain[-1] < -steel_strain_limit:
            return STEEL_COMPRESSION_STRAIN_LIMIT
        if steel_strain_limit is not None and results.maximum_tensile_strain[-1] > steel_strain_limit:
            return STEEL_TENSILE_STRAIN_LIMIT

    return None


def adapt_step_factor(step_factor, base_factor, recovered_div=None,
                      growth=2.0, min_ratio=1e-6):
    """
    Step factor for the next analysis increment.

    When an increment will not converge, the analysis retries it with a
    smaller step. This decides what the increment after that should use.

    - The last increment only converged after its step was divided by
      recovered_div. Stay small, because the full step just failed and
      retrying it would fail the same way. The step is not allowed below
      min_ratio * base_factor, so it cannot shrink to nothing.
    - The last increment converged on its own. Multiply the step by growth
      so it climbs back toward base_factor, the step originally asked for,
      and never past it.

    The result is a plain multiplier, not a length or a load, so the same
    number works for a DisplacementControl step or a LoadControl step.
    """
    if recovered_div:
        return max(base_factor * min_ratio, step_factor / recovered_div)
    return min(base_factor, step_factor * growth)


