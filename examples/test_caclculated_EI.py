"""
Demo of back-calculating effective flexural stiffness (EI) for a sway or
non-sway column, comparing OpenSees results against an AASHTO-based
interaction diagram.

Toggle `nonsway`/`sway` below to switch which column type is used.
"""

from libdenavit import SwayColumn2d, NonSwayColumn2d
from libdenavit.section import RC, Rectangle, ReinfRect
from math import pi, inf
import matplotlib.pyplot as plt
import numpy as np

nonsway = False
sway = True
plot_section = False

# region Input Properties
length = 11.25 * 12
fc = 4
fy = 60
rhosr = 0.02
Ab = 0.3
axis = 'x'
G_top = inf
G_bot = 1.48
EI_type = 'aci-b'
# endregion

# region Define RC section object
conc_cross_section = Rectangle(12, 12)
reinforcement = ReinfRect(10, 10, 2, 2, Ab)
section = RC(conc_cross_section, reinforcement, fc, fy, 'US')
if plot_section:
    section.plot_section()
EIeff = section.EIeff(axis, EI_type, 0)
Ec = section.Ec
print(f"{Ec= }")
print(f"{EIeff= }")
# endregion

# region Non-sway
if nonsway:
    col = NonSwayColumn2d(section, length, 1, 1, axis=axis, dxo=length / 1000)
# endregion

# region Sway definition
elif sway:
    Igc = section.EIgross(axis)
    k_top = 6 * (0.4 * Ec * Igc) / (G_top * length)
    k_bot = 6 * (0.4 * Ec * Igc) / (G_bot * length)
    col = SwayColumn2d(section, length, k_bot, k_top, 0, axis=axis, Dxo=length / 1000)
    K = col.effective_length_factor(EIeff)
    print(f"{K= }")

    col._K = K

    Pc = pi ** 2 * EIeff / (K * col.length) ** 2
    print(f'{Pc= }')
    print(f'{col.Cm= }')
    P = 66
    delta_s = 1 / (1 - P / (0.75 * Pc))
    print(f'{delta_s= }')
# endregion

# region run OpenSees interaction diagram
section_args = (1, "ElasticPP", "Concrete04_no_confinement", 20, 20)
section_kwargs = dict()

interaction_kwargs = dict(section_args=section_args, section_kwargs=section_kwargs, num_points=10)
result_ops = col.run_ops_interaction(**interaction_kwargs)
# endregion

# region run AASHTO interaction diagram
result_design = col.run_AASHTO_interaction(EI_type, section_factored=False)
# endregion

# region plot and print results
print(f"{result_ops['M1']= }")
print(f"{result_ops['P']= }")
plt.plot(result_ops["M1"], result_ops["P"], 'b-', label='ops')
plt.plot(result_design["M1"], np.array(result_design["P"]), 'r-', label='Design')
P_CS, M_CS, _ = col.section.section_interaction_2d(col.axis, 100, factored=False, only_compressive=True)
plt.plot(M_CS, P_CS, 'k-', label='Section')
plt.legend()
plt.show()
# endregion

# region calculate EI
print(f"{result_design['P']= }")
if sway:
    results_EI = col.calculated_EI_design(
        result_ops["P"], result_ops["M1"],
        P_design=result_design["P"], M2_design=result_design["M2"],
        G_bot=G_bot, G_top=G_top)
else:
    results_EI = col.calculated_EI_design(
        result_ops["P"], result_ops["M1"],
        P_design=result_design["P"], M2_design=result_design["M2"])

print(f'P: {results_EI["P"]}')
print(f'Calculated EI: {results_EI["EI_AASHTO"] if sway else results_EI["Calculated EI"]}')
print(f'EI gross: {results_EI["EIgross"]}')
# endregion
