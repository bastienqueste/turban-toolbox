import numpy as np
import pytest

from turban.process.utemp.api import (
    UTempLevel1,
    UTempLevel2,
    UTempLevel3,
    UTempLevel4,
    UTempProcessing,
)
from turban.process.utemp.config import UTempConfig
from turban.process.utemp.level4 import (
    QualityFlag,
    chi_from_shear,
    eps_from_k_batchelor,
    fit_spectrum_mle,
    k_batchelor_from_eps,
    log_likelihood,
    peak_waveno,
    theoretical_spectrum,
)
from turban.utils.seawater import diffusivity_temp, viscosity_kinematic

NU = 1.3e-6
D = 1.4e-7
Q = {"batchelor": 3.7, "kraichnan": 5.26}


def _cfg(**kwarg):
    defaults = dict(
        sampfreq=512.0,
        segment_length=1024,
        segment_overlap=512,
        chunk_length=3072,
        chunk_overlap=1536,
    )
    defaults.update(kwarg)
    return UTempConfig(**defaults)


@pytest.mark.parametrize("model", ["batchelor", "kraichnan"])
def test_theoretical_spectrum_integral_and_peak(model):
    k = np.linspace(0, 3000, 300001)
    kb = 250.0
    psi = theoretical_spectrum(model, k, kb, 1e-8, D, Q[model])
    assert 6 * D * np.trapezoid(psi, k) == pytest.approx(1e-8, rel=1e-6)
    assert k[np.argmax(psi)] == pytest.approx(peak_waveno(model, kb, Q[model]), abs=0.02)


def test_k_batchelor_roundtrip():
    eps = np.array([1e-10, 1e-7])
    kb = k_batchelor_from_eps(eps, NU, D)
    # k_B in cpm: (eps / (nu D^2))^(1/4) / (2 pi)
    assert kb[0] == pytest.approx((1e-10 / (NU * D**2)) ** 0.25 / (2 * np.pi))
    assert eps_from_k_batchelor(kb, NU, D) == pytest.approx(eps)


def test_log_likelihood_maximal_for_true_spectrum():
    rng = np.random.default_rng(1)
    expected = np.ones(400)
    obs = expected * rng.chisquare(9, size=400) / 9
    ll = [log_likelihood(obs, a * expected, 0 * expected, 9.5) for a in (0.5, 1, 2)]
    assert ll[1] > ll[0] and ll[1] > ll[2]


def _synthetic_spectrum(model, eps, chi, speed=0.35, seed=0, noise_level=1e-9):
    """Synthetic observed spectrum: theoretical + noise, chi2-distributed."""
    cfg = _cfg()
    rng = np.random.default_rng(seed)
    freq = np.linspace(0, 256, 513)
    k = freq / speed
    kb = k_batchelor_from_eps(eps, NU, D)
    noise = np.full_like(k, noise_level)
    expected = theoretical_spectrum(model, k, kb, chi, D, Q[model]) + noise
    dof = int(cfg.dof_spectrum)
    obs = expected * rng.chisquare(dof, size=k.shape) / dof
    return k, obs, noise, kb


@pytest.mark.parametrize("model", ["batchelor", "kraichnan"])
@pytest.mark.parametrize("eps, chi", [(1e-9, 1e-9), (1e-8, 1e-8), (1e-10, 1e-10)])
def test_fit_spectrum_mle_recovers_parameters(model, eps, chi):
    cfg = _cfg(spectrum_models=(model,))
    estimates = []
    for seed in range(5):
        k, obs, noise, kb = _synthetic_spectrum(model, eps, chi, seed=seed)
        res = fit_spectrum_mle(model, k, obs, noise, np.ones_like(k), 98 / 0.35, NU, D, cfg)
        estimates.append((res.chi, res.k_batchelor, res.eps))
    est = np.median(np.array(estimates), axis=0)
    assert est[0] == pytest.approx(chi, rel=0.25)
    assert est[1] == pytest.approx(kb, rel=0.15)
    assert est[2] == pytest.approx(eps, rel=0.8)


def test_fit_spectrum_mle_noise_only_gives_no_estimate():
    cfg = _cfg()
    k = np.linspace(0, 256, 513) / 0.35
    noise = np.full_like(k, 1e-6)
    res = fit_spectrum_mle("kraichnan", k, noise.copy(), noise, np.ones_like(k), 280.0, NU, D, cfg)
    assert np.isnan(res.chi)


@pytest.mark.parametrize("model", ["batchelor", "kraichnan"])
def test_chi_from_shear(model):
    cfg = _cfg()
    eps, chi = 1e-8, 1e-8
    k, obs, noise, _ = _synthetic_spectrum(model, eps, chi, seed=3)
    res = chi_from_shear(model, k, obs, noise, np.ones_like(k), 280.0, NU, D, eps, cfg)
    assert res.chi == pytest.approx(chi, rel=0.2)
    assert 0 < res.resolved_var_frac <= 1.2
    # without epsilon there is no estimate
    res = chi_from_shear(model, k, obs, noise, np.ones_like(k), 280.0, NU, D, np.nan, cfg)
    assert np.isnan(res.chi)


def test_shear_constrained_fit():
    cfg = _cfg(shear_constrained_fit=True)
    eps, chi = 1e-8, 1e-8
    k, obs, noise, kb = _synthetic_spectrum("kraichnan", eps, chi, seed=4)
    res = fit_spectrum_mle(
        "kraichnan", k, obs, noise, np.ones_like(k), 280.0, NU, D, cfg,
        eps_shear=eps, eps_shear_sigma_ln=0.3,
    )
    # search is restricted to eps * exp(+-0.6)
    assert eps * np.exp(-0.61) <= res.eps <= eps * np.exp(0.61)
    assert res.chi == pytest.approx(chi, rel=0.25)


def _synthetic_timeseries(model, eps, chi, speed, sampfreq, n, seed=0):
    """dT/dt time series whose gradient spectrum is the theoretical one."""
    rng = np.random.default_rng(seed)
    freq = np.fft.rfftfreq(n, 1 / sampfreq)
    k = freq / speed
    kb = k_batchelor_from_eps(eps, NU, D)
    psi_k = theoretical_spectrum(model, k, kb, chi, D, Q[model])
    psi_f_dtdt = psi_k * speed  # Psi_f(dT/dt) = W * Psi_k(dT/dz)
    df = freq[1] - freq[0]
    amp = np.sqrt(psi_f_dtdt * df / 4) * n
    coef = amp * (rng.normal(size=len(freq)) + 1j * rng.normal(size=len(freq)))
    coef[0] = 0
    return np.fft.irfft(coef, n)


def test_processing_end_to_end():
    sampfreq, speed, n = 512.0, 0.4, 3072 * 12
    eps, chi = 1e-9, 1e-9
    x = np.stack(
        [
            _synthetic_timeseries("kraichnan", eps, chi, speed, sampfreq, n, seed=s)
            for s in range(2)
        ]
    )
    # no turbulence in the first chunks, so that the empirical noise model finds them
    x[:, : 3 * 3072] = 0
    # white measurement noise
    x += np.random.default_rng(5).normal(scale=2e-5, size=x.shape)
    cfg = _cfg(fp07_response="none", noise_percentile=5.0)
    level1 = UTempLevel1(
        time=np.arange(n) / sampfreq,
        dtempdt=x,
        senspeed=np.full(n, speed),
        section_number=np.ones(n, dtype=int),
        cfg=cfg,
    )
    level1.add_aux_data({"temp": (["time"], np.full(n, 12.0), {"mean": "temp_mean"})})
    p = UTempProcessing(level1)
    l4 = p.level4
    assert list(l4.spectrum_model) == ["batchelor", "kraichnan"]
    assert l4.chi.shape == (2, 2, len(l4.time))
    assert l4.molvisc[0] == pytest.approx(viscosity_kinematic(12.0))
    assert l4.diffusivity[0] == pytest.approx(diffusivity_temp(12.0))
    quiet = l4.time * sampfreq + 1536 <= 3 * 3072  # chunks without turbulence
    turbulent = l4.time * sampfreq - 1536 >= 3 * 3072
    assert np.all(l4.quality_metric[:, :, quiet] & QualityFlag.NO_ESTIMATE)
    chi_k = np.nanmedian(l4.chi[1][:, turbulent])
    assert chi_k == pytest.approx(chi, rel=0.3)
    assert np.nanmedian(l4.eps[1][:, turbulent]) == pytest.approx(eps, rel=1.0)
    # shear products are not computed unless requested
    assert np.all(np.isnan(l4.chi_shear))
    assert np.all(l4.quality_metric_shear == QualityFlag.NO_ESTIMATE)
    # export works
    ds = l4.to_xarray()
    assert "chi" in ds and "spectrum_model" in ds.coords


def test_add_shear_dissipation_and_chi_from_shear():
    sampfreq, speed, n = 512.0, 0.4, 3072 * 6
    eps, chi = 1e-9, 1e-9
    x = _synthetic_timeseries("batchelor", eps, chi, speed, sampfreq, n)[np.newaxis]
    x[:, :3072] = 0
    cfg = _cfg(fp07_response="none", chi_from_shear=True, spectrum_models=("batchelor",))
    level1 = UTempLevel1(
        time=np.arange(n) / sampfreq,
        dtempdt=x + np.random.default_rng(2).normal(scale=2e-5, size=x.shape),
        senspeed=np.full(n, speed),
        section_number=np.ones(n, dtype=int),
        cfg=cfg,
    )
    l3 = UTempLevel3.from_level_below(UTempLevel2.from_level_below(level1))
    ntime = len(l3.time)
    with pytest.raises(ValueError):
        l3.add_shear_dissipation(np.full((2, ntime + 1), eps))
    # second probe fails quality control and is ignored
    l3.add_shear_dissipation(
        np.stack([np.full(ntime, eps), np.full(ntime, 1e-5)]),
        quality_metric=np.stack([np.zeros(ntime, int), np.ones(ntime, int)]),
    )
    assert l3._aux_data["eps_shear"][1] == pytest.approx(np.full(ntime, eps))
    l4 = UTempLevel4.from_level_below(l3)
    turbulent = l4.time > 2 * 3072 / sampfreq
    assert np.nanmedian(l4.chi_shear[0, 0, turbulent]) == pytest.approx(chi, rel=0.3)
