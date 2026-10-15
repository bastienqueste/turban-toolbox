from typing import Literal

from pydantic import BaseModel

from turban.process.generic.config import SegmentConfig

SpectrumModel = Literal["batchelor", "kraichnan"]
FP07ResponseModel = Literal[
    "nash99", "rsi", "gm", "koc", "nash02", "sommer", "none", "custom"
]


class ThermistorElectronics(BaseModel):
    """Calibration and electronics of one FP07 thermistor channel (pre-emphasised).

    The calibration coefficients (`a`, `b`, `adc_fs`, `adc_bits`, `g`, `e_b`,
    `t_0`, `beta_1`, `beta_2`) are those of the Rockland Scientific
    configuration string. The remaining parameters describe the noise of the
    analogue circuit and have the defaults of the ODAS function
    ``gradT_noise_odas``.
    """

    diff_gain: float  # [s] pre-emphasis gain
    a: float
    b: float
    adc_fs: float  # [V]
    adc_bits: int
    g: float  # bridge gain
    e_b: float  # [V] bridge excitation
    t_0: float  # [K]
    beta_1: float  # [K]
    beta_2: float | None = None
    # noise model of the analogue electronics
    resistance: float = 3000.0  # [Ohm] nominal thermistor resistance R_0
    noise_voltage_1: float = 4e-9  # [V Hz-1/2] first stage amplifier E_n
    flicker_freq_1: float = 18.7  # [Hz] first stage flicker knee
    noise_voltage_2: float = 8e-9  # [V Hz-1/2] second stage amplifier E_n2
    flicker_freq_2: float = 42.0  # [Hz] second stage flicker knee
    freq_antialias: float = 100.0  # [Hz] cut-off of each 4-pole Butterworth filter
    gamma_sampler: float = 3.0  # factor relative to ideal sampler noise


class UTempConfig(SegmentConfig):
    """Configures processing of timeseries using temperature gradient spectra.

    See the manual (section *Microtemperature processing*) for the meaning of the
    individual options.
    """

    # --- Level 1 / 3: how the gradient was obtained
    gradient_method: Literal["high_pass", "first_difference"] = "high_pass"
    diff_gain: float | None = None  # [s] needed for gradient_method="first_difference"

    # --- Level 3: frequency response of the FP07 thermistor
    fp07_response: FP07ResponseModel = "nash99"
    fp07_tau0: float | None = None  # [s] time constant at 1 m/s ("custom" only)
    fp07_speed_exponent: float | None = None  # gamma in tau0 * W**gamma ("custom")
    fp07_poles: Literal[1, 2] | None = None  # ("custom" only)
    response_limit: float = 0.1  # do not use spectra where response H < this
    freq_cutoff_antialias: float | None = 98.0  # [Hz] use up to 90% of this

    # --- Level 3: noise
    noise_model: Literal["empirical", "electronics"] = "empirical"
    noise_percentile: float = 5.0  # "empirical": quietest percentage of spectra
    thermistor_electronics: list[ThermistorElectronics] | None = None

    # --- Level 4: physical properties (if not supplied as auxiliary data)
    molvisc_fallback: float = 1.6e-6  # [m2 s-1]
    diffusivity_fallback: float = 1.4e-7  # [m2 s-1]

    # --- Level 4: spectral fitting
    spectrum_models: tuple[SpectrumModel, ...] = ("batchelor", "kraichnan")
    q_batchelor: float = 3.7  # Oakey (1982)
    q_kraichnan: float = 5.26  # Bogucki et al. (1997)
    snr_limit_waveno: float = 2.0  # upper limit where Psi/Psi_noise < this (P21: 1.55)
    snr_limit_integral: float = 2.0  # fit only if chi_measured > this * chi_noise
    min_fit_points: int = 5
    ics_factor: float = 0.04  # k_star = ics_factor * Pr**-0.5 * k_B (Steinbuck 2009)
    luketina_imberger: bool = True  # raise lower limit for fine structure
    spectral_minimum_limit: bool = True  # lower upper limit to spectral minimum
    batchelor_limit: bool = False  # also limit upper wavenumber to k_B (P21)
    waveno_limit_upper: float | None = None  # [cpm] optional hard upper limit
    kb_search_eps_range: tuple[float, float] = (1e-12, 1e-5)  # [W kg-1]
    kb_search_points: int = 100
    kb_refine_points: int = 20

    # --- Level 4: use of shear-probe epsilon (aux variables `eps_shear`,
    # `eps_shear_sigma_ln`)
    chi_from_shear: bool = False  # chi with k_B fixed from shear (P21 chi_ST)
    shear_constrained_fit: bool = False  # restrict MLE search to eps_shear +- 2 sigma
    eps_shear_sigma_ln_fallback: float = 0.5  # sigma of ln(eps_shear) if not supplied

    # --- Level 4: quality control
    mad_factor: float = 2.0  # MAD_c = mad_factor * sqrt(2 / dof)
    likelihood_ratio_limit: float | None = 2.0  # log10 L/L_powerlaw; None: no test
    snr_qc_limit: float = 5.0
    eps_fit_upper_limit: float | None = None  # flag eps from fit above (P21: ~1e-7)

    @property
    def dof_spectrum(self) -> float:
        """Degrees of freedom of the spectral estimates, 1.9 N_f (Nuttall 1971)."""
        return 1.9 * self.number_fft_windows_per_chunk

    def model_q(self, model: SpectrumModel) -> float:
        return {"batchelor": self.q_batchelor, "kraichnan": self.q_kraichnan}[model]
