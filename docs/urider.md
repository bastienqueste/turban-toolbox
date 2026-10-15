# MicroRider

TURBAN reads Rockland Scientific `.p` files with `turban.instruments.microrider.rsIO.read_p_file` (requires the optional `microrider` dependencies, `pip install turban-toolbox[microrider]`). Channels are converted to physical units using the configuration string stored in the file.

## Microtemperature

`p_to_utemp_level1` builds microtemperature Level 1 data from a `.p` file. dT/dt is derived from the pre-emphasised channel (e.g. `T1_dT1`) with the high-pass method of ODAS (Mudge & Lueck 1994), using the thermistor calibration from the file. Temperature from the `T1_hires` channels is attached as auxiliary data `temp` and pressure (`P_hires`) as `pressure`.

The speed through water is not measured by the MicroRider and must be provided on the fast time base, e.g. from a glider flight model. Sections (e.g. individual dives) are marked with `section_number`; zero excludes data.

```{.python notest}
from turban.instruments.microrider.api import p_to_utemp_level1
from turban.process.utemp.api import UTempProcessing
from turban.process.utemp.config import UTempConfig

cfg = UTempConfig(
    sampfreq=512.0,
    segment_length=1024, segment_overlap=512,
    chunk_length=3072, chunk_overlap=1536,
    fp07_response="nash99",       # or "rsi"
    noise_model="electronics",    # or "empirical"
)
level1, cfg = p_to_utemp_level1("DAT_001.p", cfg, senspeed=speed, section_number=section)
level4 = UTempProcessing(level1).level4
```

With `noise_model="electronics"`, the returned `cfg` carries the thermistor circuit parameters read from the file (`cfg.thermistor_electronics`), which can be edited (for instance the anti-aliasing filter frequency `freq_antialias`) before processing.

See the [manual](manual.md#microtemperature-processing) for the processing options.
