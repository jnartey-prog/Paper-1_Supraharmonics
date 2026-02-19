# supraharmonic-aggregation

Research-grade Python package for transformer-level supraharmonic aggregation analysis in the `2-150 kHz` band.

## Quickstart

```python
import supraharmonic_aggregation as sha

config = sha.default_config()
run = sha.analyze(config)
sha.generate_artifacts(run)
```

## CLI

```bash
supraharmonic-pipeline --quickstart --output-dir manuscript/artifacts
```

## Scope
- Analytical statistics: mean, variance, RMS scaling
- Tail metrics: percentiles and exceedance probabilities
- Monte Carlo validation and benchmark comparison
- Reproducible manuscript table/figure generation

## License
MIT
