import numpy as np
import pytest

from turban.process.utemp.config import ThermistorElectronics, UTempConfig
from turban.process.utemp.level3 import (
    noise_electronics,
    noise_empirical,
    response_fp07,
    thermistor_resistance_ratio,
)

ELECTRONICS = ThermistorElectronics(
    diff_gain=0.915,
    a=-4.0,
    b=1.0,
    adc_fs=4.096,
    adc_bits=16,
    g=6.0,
    e_b=0.68,
    t_0=289.0,
    beta_1=3000.0,
)


def _cfg(**kwarg):
    return UTempConfig(
        sampfreq=512.0,
        segment_length=1024,
        segment_overlap=512,
        chunk_length=3072,
        chunk_overlap=1536,
        **kwarg,
    )


def test_noise_empirical():
    rng = np.random.default_rng(0)
    level = 10 ** rng.uniform(-3, 0, size=(2, 200))
    psi = level[..., np.newaxis] * np.ones((2, 200, 16))
    noise = noise_empirical(psi, percentile=5.0)
    assert noise.shape == (2, 16)
    # quietest 5% of spectra lie at the bottom of the distribution
    for i in range(2):
        assert noise[i, 0] <= np.percentile(level[i], 5)
        assert noise[i, 0] >= level[i].min()


def test_noise_empirical_empty():
    assert np.all(noise_empirical(np.zeros((2, 0, 8))) == 0)


@pytest.mark.parametrize(
    "model, tau, poles",
    [("nash99", 7e-3 * 0.5**-0.12, 1), ("rsi", 4.1e-3 * 0.5**-0.5, 2)],
)
def test_response_fp07(model, tau, poles):
    cfg = _cfg(fp07_response=model)
    f_c = 1 / (2 * np.pi * tau)
    h = response_fp07(np.array([0.0, f_c]), np.array([0.5]), cfg)
    assert h.shape == (1, 2)
    assert h[0, 0] == pytest.approx(1)
    assert h[0, 1] == pytest.approx(0.5**poles)


def test_response_fp07_custom_requires_parameters():
    with pytest.raises(ValueError):
        response_fp07(np.array([1.0]), np.array([0.5]), _cfg(fp07_response="custom"))
    cfg = _cfg(
        fp07_response="custom", fp07_tau0=0.01, fp07_speed_exponent=0.0, fp07_poles=1
    )
    h = response_fp07(np.array([1 / (2 * np.pi * 0.01)]), np.array([0.5]), cfg)
    assert h[0, 0] == pytest.approx(0.5)


def test_thermistor_resistance_ratio():
    # at T_0 the resistance equals R_0
    assert thermistor_resistance_ratio(289.0 - 273.15, ELECTRONICS) == pytest.approx(1)
    # resistance decreases with temperature
    r = thermistor_resistance_ratio(np.array([5.0, 25.0]), ELECTRONICS)
    assert r[0] > 1 > r[1]


def test_noise_electronics():
    freq = np.linspace(0, 256, 513)
    noise = noise_electronics(freq, np.array([10.0, 20.0]), 512.0, ELECTRONICS)
    assert noise.shape == (2, 513)
    assert np.isnan(noise[:, 0]).all()
    assert (noise[:, 1:] > 0).all()
    # differentiator: noise of dT/dt rises with frequency below the AA filter
    assert noise[0, 100] > noise[0, 10]
