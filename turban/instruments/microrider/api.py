"""Conversion of Rockland Scientific MicroRider data to TURBAN processing levels."""

from typing import Sequence

import numpy as np
from jaxtyping import Float, Int, Num
from numpy import ndarray
from scipy.signal import butter, lfilter, lfiltic

from turban.instruments.microrider.rsIO import MicroRiderData, read_p_file
from turban.process.utemp.api import UTempLevel1
from turban.process.utemp.config import ThermistorElectronics, UTempConfig
from turban.process.utemp.level3 import gradient_scale_factor
from turban.utils.logging import get_logger

logger = get_logger(__name__)


def thermistor_electronics(data: MicroRiderData, name: str) -> ThermistorElectronics:
    """Collect calibration and circuit parameters of thermistor `name` (e.g. "T1").

    Parameters are taken from the pre-emphasised channel (e.g. "T1_dT1") where
    present and from the channel without pre-emphasis otherwise, as in ODAS.
    """
    pre = data.channel_matrix.channels[f"{name}_d{name}"].config
    plain = data.channel_matrix.channels[name].config

    def get(key, required=True):
        for cfg in (pre, plain):
            if cfg.is_set(key) if hasattr(cfg, key) else False:
                return getattr(cfg, key)
        if required:
            raise ValueError(f"Parameter `{key}` of thermistor {name} not found")
        return None

    beta_1 = get("beta_1", required=False) or get("beta")
    return ThermistorElectronics(
        diff_gain=get("diff_gain"),
        a=get("a"),
        b=get("b"),
        adc_fs=get("adc_fs"),
        adc_bits=get("adc_bits"),
        g=get("g"),
        e_b=get("e_b"),
        t_0=get("t_0"),
        beta_1=beta_1,
        beta_2=get("beta_2", required=False),
    )


def temperature_derivative(
    t_dt_counts: Num[ndarray, "time"],
    temp: Float[ndarray, "time"],
    sampfreq: float,
    el: ThermistorElectronics,
) -> Float[ndarray, "time"]:
    """dT/dt [K s-1] from the raw pre-emphasised thermistor signal.

    The pre-emphasised signal x + G dx/dt is high-pass filtered with the same
    time constant G and divided by G, which leaves dx/dt in counts per second
    (Mudge & Lueck 1994). Counts are converted to Kelvin with the local
    sensitivity of the thermistor (ODAS ``make_gradT_odas``, method "high_pass").

    Parameters
    ----------
    t_dt_counts : ndarray
        Raw pre-emphasised signal (ADC counts), e.g. channel ``T1_dT1``.
    temp : ndarray
        Temperature [degC] of the same thermistor, e.g. channel ``T1_hires``.
    sampfreq : float
        Sampling frequency [Hz].
    el : ThermistorElectronics
        Calibration of the thermistor channel.
    """
    x = np.asarray(t_dt_counts, dtype=float)
    fc = 1 / (2 * np.pi * el.diff_gain)
    b, a = butter(1, fc / (sampfreq / 2), btype="high")
    zi = lfiltic(b, a, [0.0], x[:1])
    dxdt, _ = lfilter(b, a, x, zi=zi)
    dxdt /= el.diff_gain
    return gradient_scale_factor(temp, el) * dxdt


def p_to_utemp_level1(
    data: MicroRiderData | str,
    cfg: UTempConfig,
    senspeed: Float[ndarray, "time"] | float,
    thermistors: Sequence[str] = ("T1", "T2"),
    section_number: Int[ndarray, "time"] | None = None,
) -> tuple[UTempLevel1, UTempConfig]:
    """Create microtemperature Level 1 data from a MicroRider .p file.

    Parameters
    ----------
    data : MicroRiderData or str
        Data read with :func:`turban.instruments.microrider.rsIO.read_p_file`, or
        the path of a .p file.
    cfg : UTempConfig
        Processing configuration. If ``cfg.noise_model == "electronics"`` and
        ``cfg.thermistor_electronics`` is not set, it is filled from the
        configuration string of the file.
    senspeed : ndarray or float
        Speed of the sensors through water [m s-1] on the fast time base (or a
        constant). For gliders, this typically comes from a flight model.
    thermistors : sequence of str, optional
        Names of the thermistors to use. Default ("T1", "T2").
    section_number : ndarray of int, optional
        Section markers on the fast time base; zero marks data not to be used.
        Default: all data in one section.

    Returns
    -------
    level1 : UTempLevel1
        dT/dt of each thermistor, with auxiliary data ``temp`` (degC, each
        thermistor) and ``pressure``.
    cfg : UTempConfig
        The configuration used (with the thermistor electronics filled in).
    """
    if isinstance(data, str):
        data = read_p_file(data)
    fs = float(data.header.fs_fast)
    if not np.isclose(fs, cfg.sampfreq, rtol=1e-3):
        logger.warning(f"Sampling frequency of file ({fs} Hz) differs from cfg.sampfreq ({cfg.sampfreq} Hz)")

    electronics = [thermistor_electronics(data, name) for name in thermistors]
    dtempdt, temp = [], []
    for name, el in zip(thermistors, electronics):
        raw = data.channel_matrix.channels[f"{name}_d{name}"].data
        t_hires = np.asarray(data.channel_matrix.channels[f"{name}_hires"].data, dtype=float)
        temp.append(t_hires)
        dtempdt.append(temperature_derivative(raw, t_hires, fs, el))

    ntime = len(dtempdt[0])
    time = data.header.timestamp + np.arange(ntime) / fs
    if np.isscalar(senspeed):
        senspeed = np.full(ntime, float(senspeed))
    if section_number is None:
        section_number = np.ones(ntime, dtype=int)

    if cfg.noise_model == "electronics" and cfg.thermistor_electronics is None:
        cfg = cfg.model_copy(update=dict(thermistor_electronics=electronics))

    level1 = UTempLevel1(
        time=time,
        dtempdt=np.stack(dtempdt),
        senspeed=np.asarray(senspeed, dtype=float),
        section_number=np.asarray(section_number, dtype=int),
        cfg=cfg,
    )
    level1.add_aux_data(
        {"temp": (["n_temp", "time"], np.stack(temp), {"mean": "temp_mean"})}
    )
    if "P_hires" in data.channel_matrix.channels:
        pressure = data.channel_matrix.channels["P_hires"].data
        if len(pressure) == ntime:
            level1._aux_data["pressure"] = (["time"], pressure, {"mean": "pressure_mean"})
    return level1, cfg
