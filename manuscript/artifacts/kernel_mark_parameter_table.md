# Kernel and Mark Parameter Table (Code-Specified)

This table documents the explicit parameterization currently implemented in code.

## A) Main package kernel (ExponentialKernel)

| Symbol | Meaning | Value / Definition | Units | Source |
|---|---|---|---|---|
| \(r\) | Series resistance per length | `r_ohm_per_km = 0.35` | ohm/km | `src/supraharmonic_aggregation/core/kernel.py:25` |
| \(l\) | Series inductance per length | `l_h_per_km = 0.45e-3` | H/km | `src/supraharmonic_aggregation/core/kernel.py:26` |
| \(c\) | Shunt capacitance per length | `c_f_per_km = 120e-9` | F/km | `src/supraharmonic_aggregation/core/kernel.py:27` |
| \(g\) | Shunt conductance per length | `g_s_per_km = 0.0` | S/km | `src/supraharmonic_aggregation/core/kernel.py:28` |
| \(R_\mathrm{th}\) | Source/Thevenin resistance (real-only in this model path) | `source_impedance_ohm = 0.03` | ohm | `src/supraharmonic_aggregation/core/kernel.py:29` |
| \(Z_L\) | Load impedance (only when resistive termination is used) | `load_impedance_ohm = 0.30` | ohm | `src/supraharmonic_aggregation/core/kernel.py:30` |
|  | Termination mode | default `termination_mode="matched"` (uses \(Z_L=Z_c\)) | - | `src/supraharmonic_aggregation/core/kernel.py:35,81-85` |
| \(Z_c(f)\) | Characteristic impedance | \(\sqrt{Z'(f)/Y'(f)}\) where \(Z' = r_\mathrm{ac}+j\omega l,\ Y'=(g+g_\mathrm{dielectric})+j\omega c\) | ohm | `src/supraharmonic_aggregation/core/kernel.py:69-74` |
| \(\gamma(f)\) | Propagation constant | \(\sqrt{Z'(f)Y'(f)}+\alpha\) | 1/km (effective) | `src/supraharmonic_aggregation/core/kernel.py:71-72` |
| \(\alpha\) | Added attenuation constant (`kernel_alpha`) | default `0.8` | 1/km (effective) | `src/supraharmonic_aggregation/config.py:20` |
| \(f_0\) | Resonance center frequency | \(f_0=\frac{1}{2\pi\sqrt{lc}}\) | Hz | `src/supraharmonic_aggregation/core/kernel.py:52-53` |
| \(w\) | Resonance width | \(w=\max(0.2f_0,1)\) | Hz | `src/supraharmonic_aggregation/core/kernel.py:54` |
|  | Resonance scale | `resonance_scale = 0.05` | - | `src/supraharmonic_aggregation/config.py:21` |

Using default \(l=0.45\times10^{-3}\) H/km and \(c=120\times10^{-9}\) F/km:
- \(f_0 = 21658.244479\) Hz
- \(w = 4331.648896\) Hz

## B) Feeder benchmark kernel path (scripts)

| Symbol | Meaning | Value / Definition | Units | Source |
|---|---|---|---|---|
| \(r,l,c,g\) | Feeder-specific RLGC parameters | Per feeder A/B/C (see below) | mixed | `scripts/config.py:90-124` |
| \(Z_c(f)\) | Characteristic impedance | \(\sqrt{z/y}\) | ohm | `scripts/feeder_benchmark.py:62` |
| \(\gamma(f)\) | Propagation constant | \(\sqrt{zy}+1/D_0\) | 1/m (effective additive term) | `scripts/feeder_benchmark.py:61` |
| \(D_0\) | Envelope/attenuation constant | `DIST_ATTENUATION_D0_M = 50.0` | m | `scripts/config.py:61` |
| \(Z_\mathrm{th}(f)\) | Thevenin source impedance | \(R_\mathrm{th}+j\omega L_\mathrm{th}\) | ohm | `scripts/feeder_benchmark.py:63` |
| \(Z_L\) | Right-end term in benchmark | matched \(Z_c\) (`z_right_term = zc`) | ohm | `scripts/feeder_benchmark.py:64` |

Feeder-specific constants:

| Feeder | \(r\) (ohm/km) | \(l\) (H/km) | \(c\) (F/km) | \(g\) (S) | \(R_\mathrm{th}\) (ohm) | \(L_\mathrm{th}\) (H) | \(Z_L\) resistive field (ohm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.24 | \(0.38\times10^{-3}\) | \(190\times10^{-9}\) | 0.0 | 0.010 | \(65\times10^{-6}\) | 0.28 |
| B | 0.38 | \(0.52\times10^{-3}\) | \(120\times10^{-9}\) | 0.0 | 0.014 | \(105\times10^{-6}\) | 0.31 |
| C | 0.62 | \(0.78\times10^{-3}\) | \(82\times10^{-9}\) | 0.0 | 0.020 | \(150\times10^{-6}\) | 0.35 |

## C) Mark distribution parameterization

### C1) Main package Monte Carlo / analytical assumptions

| Symbol | Meaning | Value / Definition | Source |
|---|---|---|---|
| \(\rho\) | Coherence parameter | `coherence` default `0.0`, constrained to \([0,1]\) | `src/supraharmonic_aggregation/config.py:17,44-45` |
| \(I_0\) | Base current scale | `base_current_a = 1.0` | `src/supraharmonic_aggregation/config.py:18` |
| \(\sigma_{\ln I}\) | Lognormal sigma for source amplitude | \(0.35 + 0.15(1-\rho)\) | `src/supraharmonic_aggregation/core/marks.py:82` |
| \(\mu_{\ln I}\) | Lognormal mu for source amplitude | \(\ln(I_0) - 0.5\sigma_{\ln I}^2\) | `src/supraharmonic_aggregation/core/marks.py:83` |
| \(I\) sampling | Amplitude draw | `rng.lognormvariate(mu_ln, sigma_ln)` with clipping and burst tail | `src/supraharmonic_aggregation/core/marks.py:84-87` |
| \(\phi\) distribution | Phase model | von Mises, \(\kappa=0.2+26\rho\) around common phase | `src/supraharmonic_aggregation/core/marks.py:79-80` |

Note: phase is not uniform in this main model path except in the low-coherence limit where von Mises concentration is small.

### C2) Feeder simulation mark assumptions (script path)

| Symbol | Meaning | Value / Definition | Source |
|---|---|---|---|
| \(\phi\) distribution | Phase | Uniform \([0,2\pi)\) | `scripts/feeder_sim.py:126` |
| \(\mu_{\ln I}\) | Lognormal mu | `I_LOGN_MU_LN = -3.65` | `scripts/config.py:65` |
| \(\sigma_{\ln I}\) | Lognormal sigma | `I_LOGN_SIGMA_LN = 0.55` | `scripts/config.py:66` |
| \(I\) sampling | Amplitude draw | `rng.lognormal(mean=..., sigma=...)` | `scripts/feeder_sim.py:127-128` |

## D) Numerical/gating constants requested

| Parameter | Value | Source |
|---|---:|---|
| `GATE_DENOM_MIN` | `0.05` | `scripts/config.py:39` |
| `threshold` | `1.0` | `src/supraharmonic_aggregation/config.py:22` |
| `threshold_rms_multiplier` | `1.5` | `src/supraharmonic_aggregation/config.py:23` |

## E) Ambiguity resolution for `kernel_alpha`

`kernel_alpha` is used as an additive real attenuation term in \(\gamma(f)\), i.e., \(\gamma(f)=\sqrt{Z'(f)Y'(f)}+\alpha\).  
It is not a power-law exponent.

Sources:
- `src/supraharmonic_aggregation/config.py:20`
- `src/supraharmonic_aggregation/core/kernel.py:71-72`
