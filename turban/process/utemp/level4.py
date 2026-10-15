"""Level 4 of microtemperature processing: dissipation rates from spectra.

Implements, per dissipation chunk and thermistor,

1. chi (and epsilon) from a maximum likelihood fit of a theoretical temperature
   gradient spectrum (Batchelor or Kraichnan) to the observed spectrum, after
   Ruddick et al. (2000) with the modifications of Piccolroaz et al. (2021) and the
   ODAS implementation ``get_scalardiss_odas_v2`` (Rockland Scientific; A. ten
   Doeschate): chi is constrained by integrating the observed spectrum and
   corrected for unresolved variance for every Batchelor wavenumber tried, the
   integration limits are refined in three passes.
2. chi from the integrated spectrum with the Batchelor wavenumber fixed by
   epsilon from shear probes (chi_ST in Piccolroaz et al. 2021).
3. epsilon from the fitted Batchelor wavenumber.

Wavenumbers are in cpm throughout; the Batchelor wavenumber k_B is in cpm, i.e.
k_B = (epsilon / (nu D^2))^(1/4) / (2 pi).

References
----------
Ruddick, B., Anis, A., Thompson, K., 2000: Maximum likelihood spectral fitting:
the Batchelor spectrum. J. Atmos. Oceanic Technol. 17, 1541-1555.
Steinbuck, J. V., Stacey, M. T., Monismith, S. G., 2009: An evaluation of chi_T
estimation techniques: implications for Batchelor fitting and epsilon.
J. Atmos. Oceanic Technol. 26, 1652-1662.
Luketina, D. A., Imberger, J., 2001: Determining turbulent kinetic energy
dissipation from Batchelor curve fitting. J. Atmos. Oceanic Technol. 18, 100-113.
Piccolroaz, S. et al., 2021: Lake Garda turbulence dataset. Sci. Data 8, 188.
"""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float
from numpy import ndarray, newaxis
from scipy.optimize import minimize_scalar
from scipy.special import erfc
from scipy.stats import chi2

from turban.process.utemp.config import SpectrumModel, UTempConfig
from turban.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# theoretical spectra
# ---------------------------------------------------------------------------


def _shape_batchelor(y):
    return y * (np.exp(-(y**2) / 2) - y * np.sqrt(np.pi / 2) * erfc(y / np.sqrt(2)))


def _shape_kraichnan(y):
    return y * np.exp(-np.sqrt(3) * y)


_SHAPES = {"batchelor": _shape_batchelor, "kraichnan": _shape_kraichnan}


def theoretical_spectrum(
    model: SpectrumModel,
    waveno: Float[ndarray, "..."],
    k_batchelor: Float[ndarray, "..."] | float,
    chi: Float[ndarray, "..."] | float,
    diffusivity: float,
    q: float,
) -> Float[ndarray, "..."]:
    """One-dimensional temperature gradient spectrum [K2 m-2 cpm-1].

    Batchelor (1959) or Kraichnan (1968) form, written as in Sanchez et al. (2011)
    and Piccolroaz et al. (2021, eq. 13-14). Both integrate to chi / (6 D).

    Parameters
    ----------
    model : {"batchelor", "kraichnan"}
        Spectral form.
    waveno : ndarray
        Wavenumber [cpm].
    k_batchelor : ndarray or float
        Batchelor wavenumber [cpm], broadcast against `waveno`.
    chi : ndarray or float
        Dissipation rate of temperature variance [K2 s-1].
    diffusivity : float
        Molecular thermal diffusivity [m2 s-1].
    q : float
        Universal constant of the spectrum.
    """
    y = waveno * np.sqrt(2 * q) / k_batchelor
    return chi / (2 * diffusivity * k_batchelor) * np.sqrt(2 * q) * _SHAPES[model](y)


def _peak_y(model: SpectrumModel) -> float:
    """Location of the spectral peak in units of y = k sqrt(2q) / k_B."""
    if model == "kraichnan":
        return 1 / np.sqrt(3)
    res = minimize_scalar(lambda y: -_shape_batchelor(np.array(y)), bounds=(0.01, 3), method="bounded")
    return float(res.x)


_PEAK_Y = {m: _peak_y(m) for m in _SHAPES}


def peak_waveno(model: SpectrumModel, k_batchelor: float, q: float) -> float:
    """Wavenumber [cpm] of the maximum of the theoretical spectrum."""
    return _PEAK_Y[model] * k_batchelor / np.sqrt(2 * q)


def k_batchelor_from_eps(eps, molvisc: float, diffusivity: float):
    """Batchelor wavenumber [cpm] from epsilon [W kg-1]."""
    return (eps / (molvisc * diffusivity**2)) ** 0.25 / (2 * np.pi)


def eps_from_k_batchelor(k_batchelor, molvisc: float, diffusivity: float):
    """Epsilon [W kg-1] from the Batchelor wavenumber [cpm]."""
    return molvisc * diffusivity**2 * (2 * np.pi * k_batchelor) ** 4


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------


def log_likelihood(
    psi_obs: Float[ndarray, "... waveno"],
    psi_model: Float[ndarray, "... waveno"],
    psi_noise: Float[ndarray, "... waveno"],
    dof: float,
) -> Float[ndarray, "..."] | float:
    """Log-likelihood C11 of Ruddick et al. (2000, eq. 18), summed over wavenumber.

    The observed spectrum divided by the expected spectrum (model plus noise) is
    assumed chi-squared distributed with `dof` degrees of freedom.
    """
    d = np.floor(dof)
    expected = psi_model + psi_noise
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_p = np.log(d / expected) + chi2.logpdf(d * psi_obs / expected, d)
    return np.sum(np.where(np.isfinite(ln_p), ln_p, 0.0), axis=-1)


def _trapz_masked(y, x, mask):
    """Trapezoidal integral over the intervals whose both end points are in mask."""
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x)
    seg_mask = mask[..., 1:] & mask[..., :-1]
    return np.nansum(np.where(seg_mask, seg, 0.0), axis=-1)


def _fit_costs(
    model: SpectrumModel,
    k_trials: Float[ndarray, "ntrial"],
    waveno: Float[ndarray, "waveno"],
    psi: Float[ndarray, "waveno"],
    noise: Float[ndarray, "waveno"],
    i1: int,
    i2: int,
    diffusivity: float,
    q: float,
    dof: float,
) -> tuple[Float[ndarray, "ntrial"], Float[ndarray, "ntrial"]]:
    """Likelihood and chi for each trial Batchelor wavenumber.

    For every trial k_B, chi is first estimated by integrating the observed
    (noise-subtracted) spectrum from `i1` up to min(k_B, `i2`), then corrected for
    unresolved variance as chi = chi_0^2 / chi_model, where chi_model integrates the
    theoretical spectrum (with chi_0) over the same range (Piccolroaz et al. 2021,
    eq. 6-7; ODAS ``cost_T_fit``). The likelihood is evaluated over [i1, i2].
    """
    idx = np.arange(len(waveno))
    i_kb = np.searchsorted(waveno, k_trials, side="right") - 1
    upper = np.minimum(i_kb, i2)
    mask = (idx[newaxis, :] >= i1) & (idx[newaxis, :] <= upper[:, newaxis])
    chi0 = 6 * diffusivity * (
        _trapz_masked(psi[newaxis, :], waveno, mask)
        - _trapz_masked(noise[newaxis, :], waveno, mask)
    )
    model0 = theoretical_spectrum(
        model, waveno[newaxis, :], k_trials[:, newaxis], chi0[:, newaxis], diffusivity, q
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_c = chi0**2 / (6 * diffusivity * _trapz_masked(model0, waveno, mask))
    fit = slice(i1, i2 + 1)
    psi_model = theoretical_spectrum(
        model,
        waveno[newaxis, fit],
        k_trials[:, newaxis],
        chi_c[:, newaxis],
        diffusivity,
        q,
    )
    cost = log_likelihood(psi[newaxis, fit], psi_model, noise[newaxis, fit], dof)
    cost = np.where(np.isfinite(chi_c) & (chi_c > 0), cost, -np.inf)
    return cost, chi_c


def _first(cond) -> int | None:
    hits = np.flatnonzero(cond)
    return int(hits[0]) if len(hits) else None


def _last(cond) -> int | None:
    hits = np.flatnonzero(cond)
    return int(hits[-1]) if len(hits) else None


# ---------------------------------------------------------------------------
# single-spectrum algorithms
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    chi: float = np.nan
    chi_err: float = np.nan
    k_batchelor: float = np.nan
    k_batchelor_err: float = np.nan
    eps: float = np.nan
    waveno_lo: float = np.nan
    waveno_hi: float = np.nan
    resolved_var_frac: float = np.nan
    mad: float = np.nan
    likelihood_ratio: float = np.nan
    snr: float = np.nan
    psi_model: Float[ndarray, "waveno"] | None = field(default=None, repr=False)


def _initial_limits(
    waveno: Float[ndarray, "waveno"],
    psi: Float[ndarray, "waveno"],
    noise: Float[ndarray, "waveno"],
    response: Float[ndarray, "waveno"],
    waveno_antialias: float,
    cfg: UTempConfig,
) -> tuple[float, float, float]:
    """Lowest resolved wavenumber and the noise/response/anti-aliasing limit."""
    k_lo = waveno[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = psi / noise
    i_noise = _first(snr[1:] < cfg.snr_limit_waveno)
    k_noise = waveno[-1] if i_noise is None else waveno[i_noise + 1]
    i_resp = _first(response < cfg.response_limit)
    k_resp = np.inf if i_resp is None else waveno[i_resp]
    k_hi = min(k_noise, k_resp, 0.9 * waveno_antialias)
    if cfg.waveno_limit_upper is not None:
        k_hi = min(k_hi, cfg.waveno_limit_upper)
    return k_lo, k_hi, k_noise


def fit_spectrum_mle(
    model: SpectrumModel,
    waveno: Float[ndarray, "waveno"],
    psi: Float[ndarray, "waveno"],
    noise: Float[ndarray, "waveno"],
    response: Float[ndarray, "waveno"],
    waveno_antialias: float,
    molvisc: float,
    diffusivity: float,
    cfg: UTempConfig,
    eps_shear: float = np.nan,
    eps_shear_sigma_ln: float = np.nan,
) -> FitResult:
    """Maximum likelihood fit of a theoretical spectrum to one observed spectrum.

    Parameters
    ----------
    model : {"batchelor", "kraichnan"}
        Theoretical spectrum.
    waveno : ndarray, shape (waveno,)
        Wavenumber [cpm], starting at 0.
    psi, noise : ndarray, shape (waveno,)
        Observed and noise temperature gradient spectra [K2 m-2 cpm-1], both
        corrected for the thermistor response.
    response : ndarray, shape (waveno,)
        Thermistor power transfer function H.
    waveno_antialias : float
        Wavenumber equivalent of the anti-aliasing filter [cpm].
    molvisc, diffusivity : float
        Kinematic viscosity and thermal diffusivity [m2 s-1].
    cfg : UTempConfig
        Processing configuration.
    eps_shear, eps_shear_sigma_ln : float, optional
        Epsilon from shear and the standard deviation of its logarithm. Only used
        if ``cfg.shear_constrained_fit``: the search range of k_B is then
        restricted to eps_shear * exp(+-2 sigma).
    """
    res = FitResult()
    q = cfg.model_q(model)
    dof = cfg.dof_spectrum
    prandtl = molvisc / diffusivity
    nk = len(waveno)
    if nk < 4 or not np.all(np.isfinite(psi[1:])):
        return res

    use_shear = cfg.shear_constrained_fit and np.isfinite(eps_shear)
    k_lo, k_hi, k_noise = _initial_limits(
        waveno, psi, noise, response, waveno_antialias, cfg
    )
    if use_shear:
        kb_sh = k_batchelor_from_eps(eps_shear, molvisc, diffusivity)
        k_star = np.floor(cfg.ics_factor * prandtl**-0.5 * kb_sh)
        k_lo = max(k_lo, k_star)
    i1 = _first(waveno >= k_lo)
    i2 = _last(waveno <= k_hi)
    if i1 is None or i2 is None or i2 <= i1:
        return res

    fit = slice(i1, i2 + 1)
    chi_meas = 6 * diffusivity * np.trapezoid(psi[fit], waveno[fit])
    chi_noise = 6 * diffusivity * np.trapezoid(noise[fit], waveno[fit])
    if not (chi_meas > cfg.snr_limit_integral * chi_noise and i2 - i1 + 1 > cfg.min_fit_points):
        return res

    # --- trial Batchelor wavenumbers
    if use_shear:
        sig = (
            eps_shear_sigma_ln
            if np.isfinite(eps_shear_sigma_ln)
            else cfg.eps_shear_sigma_ln_fallback
        )
        kb_l = k_batchelor_from_eps(eps_shear * np.exp(-2 * sig), molvisc, diffusivity)
        kb_h = k_batchelor_from_eps(eps_shear * np.exp(2 * sig), molvisc, diffusivity)
        k_trials = np.linspace(max(kb_l, 5 * waveno[i1]), max(kb_h, 5 * waveno[i1]), cfg.kb_refine_points)
    else:
        kb_min = k_batchelor_from_eps(cfg.kb_search_eps_range[0], molvisc, diffusivity)
        kb_max = min(
            waveno_antialias,
            k_batchelor_from_eps(cfg.kb_search_eps_range[1], molvisc, diffusivity),
        )
        k_trials = np.linspace(kb_min, kb_max, cfg.kb_search_points)
    dk = k_trials[1] - k_trials[0]
    if not dk > 0:
        return res

    def best(trials, i1_, i2_):
        cost, chis = _fit_costs(model, trials, waveno, psi, noise, i1_, i2_, diffusivity, q, dof)
        i = int(np.argmax(cost))
        return trials[i], chis[i], cost[i]

    # --- pass 1: exploratory fit over the full range
    kb0, chi00, _ = best(k_trials, i1, i2)
    psi_mod0 = theoretical_spectrum(model, waveno, kb0, chi00, diffusivity, q)
    k_peak = peak_waveno(model, kb0, q)
    k_star = np.floor(cfg.ics_factor * prandtl**-0.5 * kb0)

    # lower limit: fine structure removal after Luketina & Imberger (2001)
    with np.errstate(invalid="ignore"):
        i_cross = _first(psi < psi_mod0)
    if i_cross is not None and i_cross > 0:
        if abs(psi[i_cross] - psi_mod0[i_cross]) > abs(psi[i_cross - 1] - psi_mod0[i_cross - 1]):
            i_cross -= 1
    k_lo_li = waveno[1]
    if cfg.luketina_imberger:
        i_peak = _last(waveno <= k_peak)
        i_slope = 1
        if i_peak is not None and i_cross is not None:
            slopes = np.full(i2 + 1, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                lk, ls = np.log10(waveno), np.log10(psi)
            for i in range(3, i2 + 1):
                x, y = lk[1 : i + 1], ls[1 : i + 1]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 2:
                    slopes[i] = np.polyfit(x[ok], y[ok], 1)[0]
            n = min(i2, i_peak, i_cross) + 1
            if np.any(np.isfinite(slopes[:n])):
                i_slope = int(np.nanargmin(slopes[:n]))
                v_slope = slopes[i_slope]
                if not v_slope < 0 or np.nanmean(slopes[: i_slope + 1]) > 0:
                    i_slope = 1
                sub = slopes[3 : min(i_peak, i_cross) + 1]
                if i_slope > 1 and len(sub) and np.all(sub < 0):
                    i_slope = 1
        candidates = [waveno[i_slope], k_peak]
        if i_cross is not None:
            candidates.append(waveno[i_cross])
        k_lo_li = min(candidates)
    k_lo = max(k_star, k_lo_li, waveno[1])
    i1 = _first(waveno >= k_lo)

    # upper limit: spectral minimum above the peak (cubic fit in log-log space)
    k_min_spec = k_noise
    if cfg.spectral_minimum_limit:
        above = np.flatnonzero(waveno > k_peak)
        if len(above) > 4:
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = np.log10(waveno[above]), np.log10(psi[above])
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 4:
                p = np.polyfit(x[ok], y[ok], 3)
                d1 = np.polyder(p)
                roots = np.roots(d1)
                roots = np.sort(roots[np.isreal(roots)].real)
                roots = roots[np.polyval(np.polyder(d1), roots) > 0]
                roots = roots[roots >= np.log10(np.ceil(k_peak))]
                if len(roots):
                    k_min_spec = 10 ** roots[0]
    k_hi = min(k_min_spec, k_hi)
    if cfg.batchelor_limit:
        k_hi = min(k_hi, kb0)
    i2 = _last(waveno <= k_hi)
    if i1 is None or i2 is None or i2 <= i1:
        return res

    # --- pass 2: fit with adjusted limits
    kb01, _, _ = best(k_trials, i1, i2)
    k_star = np.floor(cfg.ics_factor * prandtl**-0.5 * kb01)
    lower = kb01 - dk
    if i_cross is not None and i_cross + 1 < nk:
        lower = max(lower, waveno[i_cross + 1])
    k_trials2 = np.linspace(lower, kb01 + dk, cfg.kb_refine_points)
    i1 = _first(waveno >= max(k_star, k_lo_li, waveno[1]))
    if i1 is None or i2 - i1 < cfg.min_fit_points:
        return res

    # --- pass 3: refined search
    cost2, chis2 = _fit_costs(model, k_trials2, waveno, psi, noise, i1, i2, diffusivity, q, dof)
    i = int(np.argmax(cost2))
    ml, chi_t, kb = cost2[i], chis2[i], k_trials2[i]
    if not np.isfinite(ml):
        return res
    fit = slice(i1, i2 + 1)
    psi_mod = theoretical_spectrum(model, waveno, kb, chi_t, diffusivity, q)

    # uncertainties from the curvature of the likelihood
    dk2 = k_trials2[1] - k_trials2[0]
    cpm, chipm = _fit_costs(
        model, np.array([kb + dk2, kb - dk2]), waveno, psi, noise, i1, i2, diffusivity, q, dof
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        curv = np.sqrt(2 * ml - cpm[0] - cpm[1])
        kb_err = abs(2 * dk2 / curv)
        chi_err = abs((chipm[0] - chipm[1]) / curv)

    # quality metrics
    expected = psi_mod[fit] + noise[fit]
    ratio = psi[fit] / expected
    mad = np.nanmean(np.abs(ratio - np.nanmean(ratio)))
    chi_v = 6 * diffusivity * np.trapezoid(psi[fit], waveno[fit])
    chi_n = 6 * diffusivity * np.trapezoid(noise[fit], waveno[fit])

    lr = np.nan
    # power-law fit over the part of the fit range where the model is above noise
    # (ODAS searched from k = 0, where the model is 0, which disabled the test)
    i_end = _first(psi_mod[i1:] < cfg.snr_limit_waveno * noise[i1:])
    end = i2 if i_end is None else min(i2, i1 + i_end)
    pl = slice(i1, end + 1)
    if end - i1 + 1 > 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = np.log(waveno[pl]), np.log(psi[pl])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2:
            pp = np.polyfit(x[ok], y[ok], 1)
            with np.errstate(divide="ignore"):
                psi_pl = np.exp(np.polyval(pp, np.log(waveno)))
            ml_pl = log_likelihood(psi[fit], psi_pl[fit], noise[fit], dof)
            lr = np.log10(np.e) * (ml - ml_pl)
            if lr < -10:
                lr = np.nan

    res.chi = chi_t
    res.chi_err = chi_err
    res.k_batchelor = kb
    res.k_batchelor_err = kb_err
    res.eps = eps_from_k_batchelor(kb, molvisc, diffusivity)
    res.waveno_lo = waveno[i1]
    res.waveno_hi = waveno[i2]
    res.resolved_var_frac = (chi_v - chi_n) / chi_t
    res.mad = mad
    res.likelihood_ratio = lr
    res.snr = chi_v / chi_n
    res.psi_model = psi_mod
    return res


def chi_from_shear(
    model: SpectrumModel,
    waveno: Float[ndarray, "waveno"],
    psi: Float[ndarray, "waveno"],
    noise: Float[ndarray, "waveno"],
    response: Float[ndarray, "waveno"],
    waveno_antialias: float,
    molvisc: float,
    diffusivity: float,
    eps_shear: float,
    cfg: UTempConfig,
) -> FitResult:
    """Chi from the observed spectrum with k_B fixed by epsilon from shear.

    The noise-subtracted spectrum is integrated over [k_L, k_U] with
    k_L = max(lowest wavenumber, 0.04 Pr^-1/2 k_B) and k_U set by the noise, the
    thermistor response and the anti-aliasing filter. The integral chi_I is then
    corrected for the unresolved variance with the theoretical spectrum:
    chi = chi_I^2 / chi_ref (Piccolroaz et al. 2021, chi_ST).
    """
    res = FitResult()
    if not (np.isfinite(eps_shear) and eps_shear > 0) or not np.all(np.isfinite(psi[1:])):
        return res
    q = cfg.model_q(model)
    kb = k_batchelor_from_eps(eps_shear, molvisc, diffusivity)
    k_lo, k_hi, _ = _initial_limits(waveno, psi, noise, response, waveno_antialias, cfg)
    k_lo = max(k_lo, np.floor(cfg.ics_factor * (molvisc / diffusivity) ** -0.5 * kb))
    if cfg.batchelor_limit:
        k_hi = min(k_hi, kb)
    i1 = _first(waveno >= k_lo)
    i2 = _last(waveno <= k_hi)
    if i1 is None or i2 is None or i2 - i1 + 1 <= cfg.min_fit_points:
        return res
    fit = slice(i1, i2 + 1)
    chi_meas = 6 * diffusivity * np.trapezoid(psi[fit], waveno[fit])
    chi_noise = 6 * diffusivity * np.trapezoid(noise[fit], waveno[fit])
    if not chi_meas > cfg.snr_limit_integral * chi_noise:
        return res
    chi_i = chi_meas - chi_noise
    ref = theoretical_spectrum(model, waveno[fit], kb, chi_i, diffusivity, q)
    chi_ref = 6 * diffusivity * np.trapezoid(ref, waveno[fit])
    chi = chi_i**2 / chi_ref
    psi_mod = theoretical_spectrum(model, waveno, kb, chi, diffusivity, q)
    ratio = psi[fit] / (psi_mod[fit] + noise[fit])
    res.chi = chi
    res.k_batchelor = kb
    res.eps = eps_shear
    res.waveno_lo = waveno[i1]
    res.waveno_hi = waveno[i2]
    res.resolved_var_frac = chi_i / chi
    res.mad = np.nanmean(np.abs(ratio - np.nanmean(ratio)))
    res.snr = chi_meas / chi_noise
    res.psi_model = psi_mod
    return res


# ---------------------------------------------------------------------------
# quality control
# ---------------------------------------------------------------------------


class QualityFlag:
    """Bit flags of the `quality_metric` variables (0 means all checks passed)."""

    MAD = 1  # mean absolute deviation above mad_factor * sqrt(2 / dof)
    SNR = 2  # signal-to-noise ratio in the fit range below snr_qc_limit
    LIKELIHOOD_RATIO = 4  # fit not better than a power law (Ruddick et al. 2000)
    FIT_RANGE = 8  # Piccolroaz et al. (2021) eq. 17 not met
    EPS_LIMIT = 16  # epsilon from fit above eps_fit_upper_limit
    NO_ESTIMATE = 32  # no estimate (spectrum too noisy or limits inconsistent)


def quality_metric_fit(
    res: FitResult, model: SpectrumModel, cfg: UTempConfig
) -> int:
    if not np.isfinite(res.chi):
        return QualityFlag.NO_ESTIMATE
    flag = 0
    if not res.mad <= cfg.mad_factor * np.sqrt(2 / cfg.dof_spectrum):
        flag |= QualityFlag.MAD
    if not res.snr >= cfg.snr_qc_limit:
        flag |= QualityFlag.SNR
    if cfg.likelihood_ratio_limit is not None and not (
        res.likelihood_ratio > cfg.likelihood_ratio_limit
    ):
        flag |= QualityFlag.LIKELIHOOD_RATIO
    k_peak = peak_waveno(model, res.k_batchelor, cfg.model_q(model))
    if not (
        res.waveno_lo < k_peak
        and np.log10(res.waveno_hi / res.waveno_lo) > 0.8
        and res.waveno_hi > 2 * k_peak
    ):
        flag |= QualityFlag.FIT_RANGE
    if cfg.eps_fit_upper_limit is not None and res.eps > cfg.eps_fit_upper_limit:
        flag |= QualityFlag.EPS_LIMIT
    return flag


def quality_metric_shear(res: FitResult, cfg: UTempConfig) -> int:
    if not np.isfinite(res.chi):
        return QualityFlag.NO_ESTIMATE
    flag = 0
    if not res.mad <= cfg.mad_factor * np.sqrt(2 / cfg.dof_spectrum):
        flag |= QualityFlag.MAD
    if not res.snr >= cfg.snr_qc_limit:
        flag |= QualityFlag.SNR
    return flag


# ---------------------------------------------------------------------------
# driver over chunks and thermistors
# ---------------------------------------------------------------------------

_FIT_FIELDS = (
    "chi",
    "chi_err",
    "k_batchelor",
    "k_batchelor_err",
    "eps",
    "waveno_lo",
    "waveno_hi",
    "resolved_var_frac",
    "mad",
    "likelihood_ratio",
    "snr",
)


def temperature_dissipation(
    psi_k: Float[ndarray, "ntemp time waveno"],
    psi_k_noise: Float[ndarray, "ntemp time waveno"],
    fp07_response: Float[ndarray, "time waveno"],
    waveno: Float[ndarray, "time waveno"],
    senspeed: Float[ndarray, "time"],
    molvisc: Float[ndarray, "time"],
    diffusivity: Float[ndarray, "time"],
    cfg: UTempConfig,
    eps_shear: Float[ndarray, "time"] | None = None,
    eps_shear_sigma_ln: Float[ndarray, "time"] | None = None,
) -> dict[str, ndarray]:
    """Dissipation rates for all chunks, thermistors and selected spectral models.

    Returns
    -------
    dict
        Arrays of shape (n_model, ntemp, time) for the MLE fit (keys of
        `FitResult`, plus ``quality_metric``) and, if ``cfg.chi_from_shear``, for chi
        with k_B from shear (keys prefixed with ``shear_``). Unused entries are NaN.
    """
    models = cfg.spectrum_models
    ntemp, ntime, _ = psi_k.shape
    shape = (len(models), ntemp, ntime)
    out = {k: np.full(shape, np.nan) for k in _FIT_FIELDS}
    out["quality_metric"] = np.full(shape, QualityFlag.NO_ESTIMATE, dtype=int)
    for k in ("chi", "waveno_lo", "waveno_hi", "resolved_var_frac", "mad", "snr"):
        out[f"shear_{k}"] = np.full(shape, np.nan)
    out["shear_quality_metric"] = np.full(shape, QualityFlag.NO_ESTIMATE, dtype=int)

    if eps_shear is None:
        eps_shear = np.full(ntime, np.nan)
        if cfg.chi_from_shear or cfg.shear_constrained_fit:
            logger.warning(
                "chi_from_shear/shear_constrained_fit requested, but no auxiliary "
                "variable `eps_shear` available; ignoring"
            )
    if eps_shear_sigma_ln is None:
        eps_shear_sigma_ln = np.full(ntime, np.nan)
    waveno_aa = (
        np.full(ntime, np.inf)
        if cfg.freq_cutoff_antialias is None
        else cfg.freq_cutoff_antialias / senspeed
    )

    for t in range(ntime):
        if not (np.isfinite(senspeed[t]) and senspeed[t] > 0):
            continue
        for i in range(ntemp):
            for m, model in enumerate(models):
                args = (
                    model,
                    waveno[t],
                    psi_k[i, t],
                    psi_k_noise[i, t],
                    fp07_response[t],
                    waveno_aa[t],
                    float(molvisc[t]),
                    float(diffusivity[t]),
                )
                res = fit_spectrum_mle(
                    *args,
                    cfg=cfg,
                    eps_shear=float(eps_shear[t]),
                    eps_shear_sigma_ln=float(eps_shear_sigma_ln[t]),
                )
                for k in _FIT_FIELDS:
                    out[k][m, i, t] = getattr(res, k)
                out["quality_metric"][m, i, t] = quality_metric_fit(res, model, cfg)
                if cfg.chi_from_shear:
                    res_sh = chi_from_shear(*args, eps_shear=float(eps_shear[t]), cfg=cfg)
                    for k in ("chi", "waveno_lo", "waveno_hi", "resolved_var_frac", "mad", "snr"):
                        out[f"shear_{k}"][m, i, t] = getattr(res_sh, k)
                    out["shear_quality_metric"][m, i, t] = quality_metric_shear(res_sh, cfg)
    return out
