# Correlation Network Reorganization Detects the 1976-77 Pacific Regime Shift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19026014.svg)](https://doi.org/10.5281/zenodo.19026014)

**Author:** Drake H. Harbert (D.H.H.)
**Affiliation:** Inner Architecture LLC, Canton, OH
**ORCID:** [0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)

## Overview

This repository contains the analysis code for detecting the 1976-77 Pacific Decadal Oscillation (PDO) regime shift using correlation network reorganization metrics, compared against conventional Critical Slowing Down (CSD) indicators.

Continuous weighted Jaccard dissimilarity (WJ) on full pairwise Spearman correlation vectors detects progressive reorganization (Kendall tau = +0.51, p = 2.1 x 10^-10), confirmed by binary Jaccard topological analysis (tau = +0.47, p = 3.3 x 10^-9). CSD indicators (lag-1 autocorrelation, variance) show no signal or trends opposite to prediction on the same data.

## Key Findings

- **WJ detects the regime shift; CSD does not.** WJ shows a progressive increase in correlation network dissimilarity from the late 1950s through the 1990s. CSD AR1 and variance trend in the wrong direction or show no signal.
- **Spatial decomposition.** Binary Jaccard identifies reorganization hubs in the Kuroshio Extension (30-52N) and southeastern subtropical Pacific (28-32S), corresponding to known PDO/ENSO teleconnection pathways.
- **Symmetric restructuring.** 14,406 edges gained and 14,412 lost between baseline and post-shift periods. The architecture reorganizes; it does not degrade.
- **Robust across all parameters.** 14/14 sensitivity variants (5 baselines, 4 window sizes, 5 thresholds) show positive WJ trends, all p < 2 x 10^-9.

## Dataset

**NOAA ERSST v5** (Extended Reconstructed Sea Surface Temperature, version 5)
- Source: [NOAA PSL](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html)
- Resolution: 2 x 2 degree, monthly
- Region: Pacific basin (60N-60S, 100E-70W)
- Period: 1955-1995 (main), 1950-2000 (sensitivity)
- 1,226 ocean grid points (fundamental units), 750,925 pairwise correlations

Data is accessed automatically via OPeNDAP (no manual download required).

## Pipeline Files

| File | Description |
|------|-------------|
| `climate_tipping_wj_pipeline.py` | Main analysis: rolling WJ trajectory, permutation testing, CSD comparison, spatial decomposition, figures |
| `compute_continuous_wj.py` | Continuous weighted Jaccard on absolute correlation vectors (primary metric) |
| `climate_sensitivity_analysis.py` | Comprehensive robustness: sliding baselines, window sizes, thresholds, 1989 shift validation, bootstrap CIs, PDO index CSD reproduction |

## Reproduction

```bash
pip install -r requirements.txt
python climate_tipping_wj_pipeline.py
python compute_continuous_wj.py
python climate_sensitivity_analysis.py
```

All pipelines are self-contained. Each downloads data via OPeNDAP, computes results, and saves figures and CSVs to local output directories. Random seed = 42 for full reproducibility.

## Dependencies

See `requirements.txt`. Core: numpy, pandas, scipy, statsmodels, xarray, netCDF4, matplotlib, seaborn, cartopy.

## License

MIT
