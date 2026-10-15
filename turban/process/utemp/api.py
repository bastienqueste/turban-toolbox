from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Float, Int, Shaped
from numpy import ndarray

from turban.process.generic.api import (
    Level,
    Level1,
    Level2,
    Level3,
    Level4,
    Processing,
    TimeseriesLevel,
)
from turban.process.utemp.config import UTempConfig
from turban.process.utemp.level3 import temperature_gradient_spectra
from turban.process.utemp.level4 import temperature_dissipation
from turban.utils.logging import get_logger
from turban.utils.seawater import diffusivity_temp, viscosity_kinematic
from turban.utils.util import agg_fast_to_slow

logger = get_logger(__name__)


@dataclass(kw_only=True)
class UTempLevel1(Level1):
    """Time derivative of temperature from fast thermistors.

    `dtempdt` is dT/dt in K s-1, one row per thermistor. Instrument readers are
    responsible for deriving it from the raw (pre-emphasised) signal. Optional
    auxiliary data used further down:

    * ``temp`` (dims ``["n_temp", "time"]`` or ``["time"]``): temperature in degC,
      aggregated to ``temp_mean``; used for viscosity, diffusivity and the
      electronics noise model.
    * ``molvisc``/``diffusivity``: override the properties computed from ``temp``.
    """

    dtempdt: Float[ndarray, "ntemp time"]
    cfg: UTempConfig


@dataclass(kw_only=True)
class UTempLevel2(Level2):
    dtempdt: Float[ndarray, "ntemp time"]

    @classmethod
    def _from_level_below_kwarg(cls, data: UTempLevel1):
        """Build constructor kwargs for UTempLevel2 from UTempLevel1 data.

        Parameters
        ----------
        data : UTempLevel1
            Level 1 microtemperature data.

        Returns
        -------
        dict
            Keyword arguments to pass to the UTempLevel2 constructor.
        """
        kwarg = super()._from_level_below_kwarg(data)
        # no despiking of temperature gradients yet
        kwarg.update(dict(dtempdt=data.dtempdt, level_below=data, cfg=data.cfg))
        return kwarg


@dataclass(kw_only=True)
class UTempLevel3(Level3):
    psi_k: Float[ndarray, "ntemp time freq"]
    psi_k_noise: Float[ndarray, "ntemp time freq"]
    psi_f: Float[ndarray, "ntemp time freq"]
    fp07_response: Float[ndarray, "time freq"]

    @classmethod
    def _from_level_below_kwarg(cls, data: UTempLevel2) -> dict:
        """Build constructor kwargs for UTempLevel3 by computing temperature gradient spectra.

        Parameters
        ----------
        data : UTempLevel2
            Level 2 microtemperature data.

        Returns
        -------
        dict
            Keyword arguments to pass to the UTempLevel3 constructor.
        """
        level2 = data
        cfg = cast(UTempConfig, level2.cfg)
        kwarg = super()._from_level_below_kwarg(level2)

        temp = _temp_per_sensor(kwarg["_aux_data"], level2.dtempdt.shape[0])
        (
            waveno,
            psi_k,
            psi_f,
            psi_k_noise,
            fp07_response,
            freq,
            senspeed,
            section_number,
        ) = temperature_gradient_spectra(
            dtempdt=level2.dtempdt,
            senspeed=level2.senspeed,
            section_number=level2.section_number,
            cfg=cfg,
            temp=temp,
        )

        kwarg.update(
            dict(
                time=agg_fast_to_slow(
                    level2.time,
                    section_number_or_data_len=level2.section_number,
                    chunk_length=cfg.chunk_length,
                    chunk_overlap=cfg.chunk_overlap,
                    agg_method="take_mid",
                ),
                psi_k=psi_k,
                psi_k_noise=psi_k_noise,
                psi_f=psi_f,
                fp07_response=fp07_response,
                waveno=waveno,
                freq=freq,
                senspeed=senspeed,
                section_number=section_number,
                level_below=level2,
                cfg=cfg,
            )
        )
        return kwarg

    def add_shear_dissipation(
        self,
        eps: Float[ndarray, "nshear time"],
        quality_metric: Int[ndarray, "nshear time"] | None = None,
        log_diss_var: Float[ndarray, "nshear time"] | None = None,
    ):
        """Attach epsilon from shear probes as auxiliary data.

        Probes are combined as the geometric mean of the probes that pass quality
        control (``quality_metric == 0``). The result is stored as auxiliary
        variables ``eps_shear`` and ``eps_shear_sigma_ln`` (standard deviation of
        ln(eps), from the ATOMIX ``log_diss_var``), which are used by Level 4 if
        ``cfg.chi_from_shear`` or ``cfg.shear_constrained_fit`` is set.

        The shear estimates must be on the same chunks as this object, i.e. be
        processed with the same sampling frequency, chunk length and overlap and
        section numbers (as e.g. ``ShearLevel4.eps``).
        """
        eps = np.atleast_2d(eps)
        if eps.shape[-1] != len(self.time):
            raise ValueError(
                f"Shear epsilon has {eps.shape[-1]} chunks, temperature spectra "
                f"{len(self.time)}; process both with the same chunking"
            )
        good = np.isfinite(eps) & (eps > 0)
        if quality_metric is not None:
            good &= np.atleast_2d(quality_metric) == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_eps = np.where(good, np.log(np.where(eps > 0, eps, np.nan)), np.nan)
            n_good = good.sum(axis=0)
            eps_comb = np.exp(np.nansum(log_eps, axis=0) / n_good)
        eps_comb = np.where(n_good > 0, eps_comb, np.nan)
        if log_diss_var is None:
            sigma = np.full(eps.shape[-1], np.nan)
        else:
            var = np.where(good, np.atleast_2d(log_diss_var), np.nan)
            with np.errstate(invalid="ignore"):
                sigma = np.sqrt(np.nanmean(var, axis=0))
        self.add_aux_data(eps_comb, "eps_shear")
        self.add_aux_data(sigma, "eps_shear_sigma_ln")


def _temp_per_sensor(aux: dict, ntemp: int) -> Float[ndarray, "ntemp time"] | None:
    """Chunk-mean temperature of each thermistor from auxiliary data, if any."""
    if "temp_mean" not in aux:
        return None
    temp = np.asarray(aux["temp_mean"][1], dtype=float)
    if temp.ndim == 1:
        temp = np.broadcast_to(temp, (ntemp, len(temp)))
    return temp


@dataclass(kw_only=True)
class UTempLevel4(Level4):
    spectrum_model: Shaped[ndarray, "n_model"]
    chi: Float[ndarray, "n_model ntemp time"]
    chi_err: Float[ndarray, "n_model ntemp time"]
    eps: Float[ndarray, "n_model ntemp time"]
    k_batchelor: Float[ndarray, "n_model ntemp time"]
    k_batchelor_err: Float[ndarray, "n_model ntemp time"]
    waveno_lo: Float[ndarray, "n_model ntemp time"]
    waveno_hi: Float[ndarray, "n_model ntemp time"]
    resolved_var_frac: Float[ndarray, "n_model ntemp time"]
    mad: Float[ndarray, "n_model ntemp time"]
    likelihood_ratio: Float[ndarray, "n_model ntemp time"]
    snr: Float[ndarray, "n_model ntemp time"]
    quality_metric: Int[ndarray, "n_model ntemp time"]
    chi_shear: Float[ndarray, "n_model ntemp time"]
    mad_shear: Float[ndarray, "n_model ntemp time"]
    snr_shear: Float[ndarray, "n_model ntemp time"]
    resolved_var_frac_shear: Float[ndarray, "n_model ntemp time"]
    quality_metric_shear: Int[ndarray, "n_model ntemp time"]
    molvisc: Float[ndarray, "time"]
    diffusivity: Float[ndarray, "time"]
    _coords = ["time", "spectrum_model"]

    @classmethod
    def _from_level_below_kwarg(cls, level3: UTempLevel3) -> dict:
        cfg = cast(UTempConfig, level3.cfg)
        kwarg = super()._from_level_below_kwarg(level3)
        aux = cast(dict, level3._aux_data)
        ntime = len(level3.time)

        temp = _temp_per_sensor(aux, level3.psi_k.shape[0])
        if "molvisc" in aux:
            molvisc = np.asarray(aux["molvisc"][1], dtype=float)
        elif temp is not None:
            molvisc = viscosity_kinematic(np.mean(temp, axis=0))
        else:
            molvisc = np.full(ntime, cfg.molvisc_fallback)
        if "diffusivity" in aux:
            diffusivity = np.asarray(aux["diffusivity"][1], dtype=float)
        elif temp is not None:
            diffusivity = diffusivity_temp(np.mean(temp, axis=0))
        else:
            diffusivity = np.full(ntime, cfg.diffusivity_fallback)

        eps_shear = aux.get("eps_shear", (None, None))[1]
        eps_shear_sigma_ln = aux.get("eps_shear_sigma_ln", (None, None))[1]

        out = temperature_dissipation(
            psi_k=level3.psi_k,
            psi_k_noise=level3.psi_k_noise,
            fp07_response=level3.fp07_response,
            waveno=level3.waveno,
            senspeed=level3.senspeed,
            molvisc=molvisc,
            diffusivity=diffusivity,
            cfg=cfg,
            eps_shear=eps_shear,
            eps_shear_sigma_ln=eps_shear_sigma_ln,
        )
        kwarg.update(
            dict(
                spectrum_model=np.array(cfg.spectrum_models),
                chi=out["chi"],
                chi_err=out["chi_err"],
                eps=out["eps"],
                k_batchelor=out["k_batchelor"],
                k_batchelor_err=out["k_batchelor_err"],
                waveno_lo=out["waveno_lo"],
                waveno_hi=out["waveno_hi"],
                resolved_var_frac=out["resolved_var_frac"],
                mad=out["mad"],
                likelihood_ratio=out["likelihood_ratio"],
                snr=out["snr"],
                quality_metric=out["quality_metric"],
                chi_shear=out["shear_chi"],
                mad_shear=out["shear_mad"],
                snr_shear=out["shear_snr"],
                resolved_var_frac_shear=out["shear_resolved_var_frac"],
                quality_metric_shear=out["shear_quality_metric"],
                molvisc=molvisc,
                diffusivity=diffusivity,
                level_below=level3,
                cfg=cfg,
            )
        )
        return kwarg


class UTempProcessing(Processing):

    _level_mapping: dict[Level, type[TimeseriesLevel]] = {
        Level.L1: UTempLevel1,
        Level.L2: UTempLevel2,
        Level.L3: UTempLevel3,
        Level.L4: UTempLevel4,
    }
