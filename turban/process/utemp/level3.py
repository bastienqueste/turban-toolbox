"""Level 3 of microtemperature processing: temperature gradient spectra.

Starting from the time derivative of temperature (K s-1), this computes per
dissipation chunk

* the frequency spectrum of dT/dt,
* the wavenumber spectrum of the temperature gradient dT/dz, corrected for the
  frequency response of the FP07 thermistor,
* the noise spectrum, boosted by the same response correction.
"""

import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray, newaxis
from scipy.signal import butter, freqz

from turban.process.utemp.config import ThermistorElectronics, UTempConfig
from turban.utils.logging import get_logger
from turban.utils.spectra import spectrum
from turban.utils.util import agg_fast_to_slow, get_chunking_index

logger = get_logger(__name__)

# (tau0 [s], exponent of speed, number of poles) of published FP07 response models.
# tau = tau0 * (W / 1 m s-1) ** exponent
FP07_RESPONSE_MODELS: dict[str, tuple[float, float, int]] = {
    "nash99": (7e-3, -0.12, 1),  # Nash et al. (1999), as in Piccolroaz et al. (2021)
    "rsi": (4.1e-3, -0.5, 2),  # Vachon & Lueck (1984), RSI default (F0 = 25 W^0.5)
    "gm": (10.6e-3, -0.32, 2),  # Gregg & Meagher (1980)
    "koc": (7e-3, -0.32, 2),  # Kocsis et al. (1999)
    "nash02": (8.5e-3, 0.0, 2),  # Nash et al. (2002) according to Sommer et al. (2013)
    "sommer": (10e-3, 0.0, 2),  # Sommer et al. (2013)
}


def temperature_gradient_spectra(
    dtempdt: Float[ndarray, "ntemp time_fast"],
    senspeed: Float[ndarray, "time_fast"],
    section_number: Int[ndarray, "time_fast"],
    cfg: UTempConfig,
    temp: Float[ndarray, "ntemp time_slow"] | None = None,
) -> tuple[
    Float[ndarray, "time_slow freq"],  # waveno
    Float[ndarray, "ntemp time_slow freq"],  # psi_k (response corrected)
    Float[ndarray, "ntemp time_slow freq"],  # psi_f (dT/dt, uncorrected)
    Float[ndarray, "ntemp time_slow freq"],  # psi_k_noise (response corrected)
    Float[ndarray, "time_slow freq"],  # fp07 response H
    Float[ndarray, "freq"],  # freq
    Float[ndarray, "time_slow"],  # senspeed
    Int[ndarray, "time_slow"],  # section_number
]:
    """Compute temperature gradient wavenumber spectra and noise spectra.

    Parameters
    ----------
    dtempdt : ndarray, shape (ntemp, time_fast)
        Time derivative of temperature [K s-1] for each thermistor.
    senspeed : ndarray, shape (time_fast,)
        Platform speed through water [m s-1].
    section_number : ndarray of int, shape (time_fast,)
        Section marker array; zero marks invalid data.
    cfg : UTempConfig
        Processing configuration.
    temp : ndarray, shape (ntemp, time_slow), optional
        Chunk-mean temperature [degC] of each thermistor. Required for
        ``cfg.noise_model == "electronics"``.

    Returns
    -------
    waveno : ndarray, shape (time_slow, freq)
        Wavenumber [cpm] for each chunk.
    psi_k : ndarray, shape (ntemp, time_slow, freq)
        Temperature gradient spectrum [K2 m-2 cpm-1], corrected for the
        frequency response of the thermistor.
    psi_f : ndarray, shape (ntemp, time_slow, freq)
        Frequency spectrum of dT/dt [K2 s-2 Hz-1], not corrected for the
        thermistor response.
    psi_k_noise : ndarray, shape (ntemp, time_slow, freq)
        Noise spectrum [K2 m-2 cpm-1], corrected in the same way as `psi_k`.
    fp07_response : ndarray, shape (time_slow, freq)
        Power transfer function H of the thermistor.
    freq : ndarray, shape (freq,)
        Frequency [Hz].
    senspeed : ndarray, shape (time_slow,)
        Chunk-averaged platform speed.
    section_number : ndarray of int, shape (time_slow,)
        Section markers aggregated to slow time.
    """
    ii = get_chunking_index(section_number, (cfg.chunk_length, cfg.chunk_overlap))

    psi_f, freq = spectrum(
        dtempdt,
        cfg.sampfreq,
        reshape_index=ii,
        segment_length=cfg.segment_length,
        segment_overlap=cfg.segment_overlap,
    )

    if cfg.gradient_method == "first_difference":
        if cfg.diff_gain is None:
            raise ValueError("gradient_method='first_difference' requires diff_gain")
        psi_f = psi_f * correction_first_difference(freq, cfg.sampfreq, cfg.diff_gain)

    senspeed_slow: Float[ndarray, "time_slow"] = agg_fast_to_slow(
        senspeed, reshape_index=ii
    )
    section_number_slow: Int[ndarray, "time_slow"] = agg_fast_to_slow(
        section_number, reshape_index=ii, agg_method="max"
    )

    fp07_response = response_fp07(freq, senspeed_slow, cfg)

    # noise spectrum of dT/dt in the frequency domain
    match cfg.noise_model:
        case "empirical":
            noise_f = noise_empirical(psi_f, cfg.noise_percentile)[:, newaxis, :]
            noise_f = np.broadcast_to(noise_f, psi_f.shape)
        case "electronics":
            if cfg.thermistor_electronics is None or temp is None:
                raise ValueError(
                    "noise_model='electronics' requires cfg.thermistor_electronics "
                    "and the chunk-mean temperature of each thermistor (aux `temp_mean`)"
                )
            noise_f = np.stack(
                [
                    noise_electronics(freq, temp[i], cfg.sampfreq, el)
                    for i, el in enumerate(cfg.thermistor_electronics)
                ]
            )
        case _:
            raise ValueError(f"Unknown noise model {cfg.noise_model}")

    # frequency spectrum of dT/dt -> wavenumber spectrum of dT/dz:
    # dT/dz = (dT/dt) / W and dk = df / W, so Psi_k = Psi_f / W
    waveno = freq[newaxis, :] / senspeed_slow[:, newaxis]
    to_waveno = 1 / senspeed_slow[newaxis, :, newaxis]
    psi_k = psi_f * to_waveno / fp07_response[newaxis, ...]
    psi_k_noise = noise_f * to_waveno / fp07_response[newaxis, ...]

    return (
        waveno,
        psi_k,
        psi_f,
        psi_k_noise,
        fp07_response,
        freq,
        senspeed_slow,
        section_number_slow,
    )


def fp07_time_constant(
    senspeed: Float[ndarray, "time"], cfg: UTempConfig
) -> tuple[Float[ndarray, "time"], int]:
    """Time constant [s] and number of poles of the FP07 frequency response."""
    if cfg.fp07_response == "custom":
        if None in (cfg.fp07_tau0, cfg.fp07_speed_exponent, cfg.fp07_poles):
            raise ValueError(
                "fp07_response='custom' requires fp07_tau0, fp07_speed_exponent "
                "and fp07_poles"
            )
        tau0, gamma, poles = cfg.fp07_tau0, cfg.fp07_speed_exponent, cfg.fp07_poles
    else:
        tau0, gamma, poles = FP07_RESPONSE_MODELS[cfg.fp07_response]
    return tau0 * np.abs(senspeed) ** gamma, int(poles)


def response_fp07(
    freq: Float[ndarray, "freq"],
    senspeed: Float[ndarray, "time"],
    cfg: UTempConfig,
) -> Float[ndarray, "time freq"]:
    """Power transfer function H(f) of the FP07 thermistor.

    H = (1 + (2 pi tau f)^2)^-n with n = 1 (single pole) or 2 (double pole), and
    tau = tau0 W^gamma. ``fp07_response='none'`` gives H = 1.
    """
    if cfg.fp07_response == "none":
        return np.ones((len(senspeed), len(freq)))
    tau, poles = fp07_time_constant(senspeed, cfg)
    return (1 + (2 * np.pi * tau[:, newaxis] * freq[newaxis, :]) ** 2) ** (-poles)


def correction_first_difference(
    freq: Float[ndarray, "freq"], sampfreq: float, diff_gain: float
) -> Float[ndarray, "freq"]:
    """Spectral correction for a gradient obtained by first-differencing.

    Combines the correction of the first-difference operator (sinc) with the
    correction for the bilinear transformation of the deconvolution filter, as in
    the ODAS function ``get_scalar_spectra_odas``.
    """
    correction = np.ones_like(freq)
    f = freq[1:]
    correction[1:] = ((np.pi * f) / (sampfreq * np.sin(np.pi * f / sampfreq))) ** 2

    b, a = butter(1, 1 / (2 * np.pi * diff_gain * sampfreq / 2))
    _, h = freqz(b, a, freq[:-1], fs=sampfreq)
    applied = np.abs(h) ** 2
    intended = 1 / (1 + (2 * np.pi * freq[:-1] * diff_gain) ** 2)
    bl_correction = np.ones_like(freq)
    bl_correction[:-1] = intended / applied
    bl_correction[-1] = bl_correction[-2]  # infinite at Nyquist
    return correction * bl_correction


def noise_empirical(
    psi_f: Float[ndarray, "ntemp time freq"], percentile: float = 5.0
) -> Float[ndarray, "ntemp freq"]:
    """Noise spectrum as geometric mean of the least energetic spectra.

    Spectra are ranked by their variance (mean level over all non-zero
    frequencies), separately for each thermistor; the spectra in the lowest
    `percentile` are averaged in log space. Works on frequency spectra, so that
    the noise of the electronics is compared at equal frequencies.
    """
    ntemp, ntime, nfreq = psi_f.shape
    noise = np.full((ntemp, nfreq), np.nan)
    if ntime == 0:
        return np.zeros((ntemp, nfreq))
    level = np.nanmean(psi_f[..., 1:], axis=-1)
    for i in range(ntemp):
        good = np.isfinite(level[i]) & (level[i] > 0)
        if not np.any(good):
            continue
        quiet = good & (level[i] <= np.nanpercentile(level[i, good], percentile))
        with np.errstate(divide="ignore"):
            noise[i] = np.exp(np.nanmean(np.log(psi_f[i, quiet, :]), axis=0))
    return noise


def thermistor_resistance_ratio(
    temp: Float[ndarray, "..."] | float, el: ThermistorElectronics
) -> Float[ndarray, "..."] | float:
    """Invert the Steinhart-Hart type calibration for the ratio R_T / R_0."""
    inv_t = 1 / (np.asarray(temp, dtype=float) + 273.15) - 1 / el.t_0
    if el.beta_2 is None:
        log_r = inv_t * el.beta_1
    else:
        # inv_t = log_r / beta_1 + log_r**2 / beta_2
        a, b = 1 / el.beta_2, 1 / el.beta_1
        log_r = (-b + np.sqrt(b**2 + 4 * a * inv_t)) / (2 * a)
    return np.exp(log_r)


def gradient_scale_factor(
    temp: Float[ndarray, "..."] | float, el: ThermistorElectronics
) -> Float[ndarray, "..."] | float:
    """dT / d(counts) of the thermistor channel, i.e. counts -> K.

    Matches the high-pass scaling in the ODAS function ``make_gradT_odas``.
    """
    r = thermistor_resistance_ratio(temp, el)
    tk = np.asarray(temp, dtype=float) + 273.15
    eta = (el.b / 2) * 2**el.adc_bits * el.g * el.e_b / el.adc_fs
    scale = np.ones_like(tk)
    if el.beta_2 is not None:
        scale = 1 + 2 * (el.beta_1 / el.beta_2) * np.log(r)
    return scale * tk**2 * (1 + r) ** 2 / (2 * eta * el.beta_1 * r)


def noise_electronics(
    freq: Float[ndarray, "freq"],
    temp: Float[ndarray, "time"],
    sampfreq: float,
    el: ThermistorElectronics,
) -> Float[ndarray, "time freq"]:
    """Noise spectrum of dT/dt [K2 s-2 Hz-1] from the RSI electronics model.

    Port of the ODAS functions ``noise_thermchannel`` and ``gradT_noise_odas``
    (RSI Technical Note 040). The noise in ADC counts is converted to dT/dt with
    the high-pass differentiator and the thermistor sensitivity at the chunk-mean
    temperature.
    """
    k_boltzmann = 1.382e-23
    f = np.where(freq > 0, freq, np.nan)[newaxis, :]
    delta_s = el.adc_fs / 2**el.adc_bits
    nyquist = sampfreq / 2
    temp_k = (np.asarray(temp, dtype=float) + 273.15)[:, newaxis]
    # Johnson noise of the thermistor at its actual resistance and temperature
    resistance = el.resistance * thermistor_resistance_ratio(temp, el)[:, newaxis]
    phi_r = 4 * k_boltzmann * resistance * temp_k
    v1 = 2 * el.noise_voltage_1**2 * np.sqrt(1 + (f / el.flicker_freq_1) ** 2) / (
        f / el.flicker_freq_1
    )
    noise_1 = el.g**2 * (v1 + phi_r)
    g_2 = 1 + (2 * np.pi * el.diff_gain * f) ** 2
    v2 = 2 * el.noise_voltage_2**2 * np.sqrt(1 + (f / el.flicker_freq_2) ** 2) / (
        f / el.flicker_freq_2
    )
    noise_2 = g_2 * (noise_1 + v2)
    g_aa = 1 / (1 + (f / el.freq_antialias) ** 8) ** 2
    noise_v = noise_2 * g_aa + el.gamma_sampler * delta_s**2 / (12 * nyquist)
    noise_counts = noise_v / delta_s**2

    # high-pass differentiator applied to the pre-emphasised signal
    g_d = el.diff_gain
    g_hp = (1 / g_d) ** 2 * (2 * np.pi * g_d * f) ** 2 / (1 + (2 * np.pi * g_d * f) ** 2)
    scale = gradient_scale_factor(temp, el)[:, newaxis]
    out = noise_counts * g_hp * scale**2
    out[:, ~np.isfinite(f[0])] = np.nan
    return out
