"""Molecular properties of seawater needed for scalar dissipation estimates.

The functions follow the formulations used in the ODAS MATLAB library, so that
results can be compared one-to-one with that toolbox:

* kinematic viscosity: polynomial fit for salinity 35 (ODAS ``visc35``)
* thermal diffusivity: k / (rho * c_p) with the seawater thermophysical
  property correlations of Sharqawy, Lienhard & Zubair (2010),
  Desalination and Water Treatment 16, 354-380.
"""

import numpy as np
from jaxtyping import Float
from numpy import ndarray

ArrayOrFloat = Float[ndarray, "..."] | float


def viscosity_kinematic(temp: ArrayOrFloat) -> ArrayOrFloat:
    """Kinematic viscosity of seawater at salinity 35.

    Parameters
    ----------
    temp : ndarray or float
        Temperature in degrees Celsius.

    Returns
    -------
    ndarray or float
        Kinematic viscosity in m2 s-1.
    """
    pol = [
        -1.131311019739306e-011,
        1.199552027472192e-009,
        -5.864346822839289e-008,
        1.828297985908266e-006,
    ]
    return np.polyval(pol, temp)


def _thermal_conductivity(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Thermal conductivity [W m-1 K-1] (Sharqawy et al. 2010, eq. 13)."""
    t68 = 1.00024 * np.asarray(temp)
    sp = np.asarray(sal) / 1.00472
    tk = t68 + 273.15
    return 10 ** (
        np.log10(240 + 0.0002 * sp)
        + 0.434 * (2.3 - (343.5 + 0.037 * sp) / tk) * (1 - tk / (647.3 + 0.03 * sp)) ** (1 / 3)
        - 3
    )


def _specific_heat(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Specific heat capacity [J kg-1 K-1] (Sharqawy et al. 2010, eq. 9)."""
    t68 = 1.00024 * np.asarray(temp)
    sp = np.asarray(sal) / 1.00472
    a = 4206.8 - 6.6197 * sp + 1.2288e-2 * sp**2
    b = -1.1262 + 5.4178e-2 * sp - 2.2719e-4 * sp**2
    c = 1.2026e-2 - 5.3566e-4 * sp + 1.8906e-6 * sp**2
    d = 6.8777e-7 + 1.517e-6 * sp - 4.4268e-9 * sp**2
    return a + b * t68 + c * t68**2 + d * t68**3


def _density(temp: ArrayOrFloat, sal: ArrayOrFloat) -> ArrayOrFloat:
    """Density [kg m-3] at atmospheric pressure (Sharqawy et al. 2010, eq. 8)."""
    t = np.asarray(temp)
    s = np.asarray(sal) / 1000
    rho_w = (
        9.9992293295e02
        + 2.0341179217e-02 * t
        - 6.1624591598e-03 * t**2
        + 2.2614664708e-05 * t**3
        - 4.6570659168e-08 * t**4
    )
    d_rho = (
        8.0200240891e02 * s
        - 2.0005183488e00 * s * t
        + 1.6771024982e-02 * s * t**2
        - 3.0600536746e-05 * s * t**3
        - 1.6132224742e-05 * s**2 * t**2
    )
    return rho_w + d_rho


def diffusivity_temp(temp: ArrayOrFloat, sal: ArrayOrFloat = 35.0) -> ArrayOrFloat:
    """Molecular thermal diffusivity of seawater.

    Parameters
    ----------
    temp : ndarray or float
        Temperature in degrees Celsius.
    sal : ndarray or float, optional
        Practical salinity. Default is 35, as used in ODAS.

    Returns
    -------
    ndarray or float
        Thermal diffusivity in m2 s-1.
    """
    return _thermal_conductivity(temp, sal) / (
        _density(temp, sal) * _specific_heat(temp, sal)
    )
