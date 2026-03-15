"""
Pipeline: Compute Continuous Weighted Jaccard on Pacific SST Correlation Vectors
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-14
Description:
    Computes continuous weighted Jaccard (WJ) on absolute correlation vectors
    as the primary analysis metric. WJ_continuous = 1 - sum(min(|a|,|b|))/sum(max(|a|,|b|))
    where a and b are flattened upper triangles of correlation matrices.
    This complements the binary Jaccard topological analysis from the main pipeline.
Dependencies: xarray, netCDF4, numpy, pandas, scipy
Input: ERSST v5 via OPeNDAP
Output: G:/My Drive/inner_architecture_research/climate_tipping_wj/results/
"""

import os
import time
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

FORCE_RECOMPUTE = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"G:\My Drive\inner_architecture_research\climate_tipping_wj"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ERSST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"

print("=" * 60)
print("CONTINUOUS WJ COMPUTATION")
print("=" * 60)

# Load data (same as main pipeline)
t0 = time.time()
local_file = os.path.join(BASE_DIR, "data", "sst.mnmean.nc")
try:
    ds = xr.open_dataset(ERSST_URL, engine='netcdf4')
except Exception:
    print("  OPeNDAP failed, using local file...")
    ds = xr.open_dataset(local_file)
sst = ds['sst'].sel(
    time=slice('1955-01', '1995-12'),
    lat=slice(60.0, -60.0),
    lon=slice(100.0, 290.0)
).load()
sst = sst.isel(lat=slice(None, None, 2), lon=slice(None, None, 2))

n_time = sst.shape[0]
print(f"  SST shape: {sst.shape}")
sst_2d = sst.values.reshape(n_time, -1)
print(f"  Flattened: {sst_2d.shape}")
nan_cols = np.all(np.isnan(sst_2d), axis=0)
print(f"  All-NaN columns: {nan_cols.sum()}")
valid_mask = ~nan_cols
sst_valid = sst_2d[:, valid_mask].copy()
print(f"  After filter: {sst_valid.shape}")

for col in range(sst_valid.shape[1]):
    cd = sst_valid[:, col]
    nm = np.isnan(cd)
    if nm.any() and not nm.all():
        sst_valid[nm, col] = np.nanmean(cd)

# Verify no NaN remaining
remaining_nan = np.isnan(sst_valid).sum()
print(f"  Remaining NaN after fill: {remaining_nan}")

time_index = pd.DatetimeIndex(sst.time.values)
n_gp = sst_valid.shape[1]
print(f"Grid points: {n_gp}, Load time: {time.time()-t0:.1f}s")


def compute_spearman_matrix(data):
    ranked = np.apply_along_axis(stats.rankdata, 0, data)
    corr = np.corrcoef(ranked, rowbar=False) if False else np.corrcoef(ranked, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    return corr


def continuous_wj(corr_A, corr_B):
    """Weighted Jaccard on absolute correlation vectors (continuous, no binarization)."""
    idx = np.triu_indices_from(corr_A, k=1)
    a = np.abs(corr_A[idx])
    b = np.abs(corr_B[idx])
    numerator = np.sum(np.minimum(a, b))
    denominator = np.sum(np.maximum(a, b))
    if denominator == 0:
        return 0.0
    return 1.0 - (numerator / denominator)


def binary_wj(corr_A, corr_B, percentile=95):
    """Binary Jaccard after binarization at percentile threshold."""
    idx = np.triu_indices_from(corr_A, k=1)
    a_vals = np.abs(corr_A[idx])
    b_vals = np.abs(corr_B[idx])
    thresh_a = np.nanpercentile(a_vals, percentile)
    thresh_b = np.nanpercentile(b_vals, percentile)
    a_bin = (a_vals >= thresh_a).astype(int)
    b_bin = (b_vals >= thresh_b).astype(int)
    intersection = np.sum(a_bin & b_bin)
    union = np.sum(a_bin | b_bin)
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


# Compute both trajectories
print("\nComputing rolling WJ trajectories (continuous + binary)...")

window_months = 60  # 5 years
step = 6
baseline_mask = (time_index.year >= 1955) & (time_index.year <= 1964)
baseline_data = sst_valid[baseline_mask, :]
baseline_corr = compute_spearman_matrix(baseline_data)

window_starts = list(range(0, n_time - window_months + 1, step))
n_windows = len(window_starts)

cwj_vals = []
bwj_vals = []
wj_times = []

t0 = time.time()
for i, ws in enumerate(window_starts):
    we = ws + window_months
    window_data = sst_valid[ws:we, :]
    center = time_index[ws] + (time_index[we-1] - time_index[ws]) / 2
    corr = compute_spearman_matrix(window_data)

    cwj = continuous_wj(baseline_corr, corr)
    bwj = binary_wj(baseline_corr, corr, 95)

    cwj_vals.append(cwj)
    bwj_vals.append(bwj)
    wj_times.append(center)

    if (i+1) % 20 == 0:
        print(f"  Window {i+1}/{n_windows}")

cwj_vals = np.array(cwj_vals)
bwj_vals = np.array(bwj_vals)
print(f"  Time: {time.time()-t0:.1f}s")

# Kendall tau
def kendall(v):
    x = np.arange(len(v))
    return stats.kendalltau(x, v)

tau_cwj, p_cwj = kendall(cwj_vals)
tau_bwj, p_bwj = kendall(bwj_vals)

# Correlation between the two metrics
corr_metrics, p_corr = stats.spearmanr(cwj_vals, bwj_vals)

print(f"\nResults:")
print(f"  Continuous WJ:  tau = {tau_cwj:.4f} (p = {p_cwj:.2e})")
print(f"  Binary J:       tau = {tau_bwj:.4f} (p = {p_bwj:.2e})")
print(f"  Correlation between metrics: rho = {corr_metrics:.4f} (p = {p_corr:.2e})")
print(f"  Continuous WJ range: {cwj_vals.min():.4f} to {cwj_vals.max():.4f}")
print(f"  Binary J range: {bwj_vals.min():.4f} to {bwj_vals.max():.4f}")

# Permutation test for continuous WJ
print("\nPermutation test (continuous WJ)...")
post_mask = (time_index.year >= 1978) & (time_index.year <= 1983)
post_data = sst_valid[post_mask, :]
post_corr = compute_spearman_matrix(post_data)
obs_cwj = continuous_wj(baseline_corr, post_corr)

pre_mask = (time_index.year >= 1965) & (time_index.year <= 1970)
pre_data = sst_valid[pre_mask, :]
pre_corr = compute_spearman_matrix(pre_data)
obs_cwj_control = continuous_wj(baseline_corr, pre_corr)

all_data = np.vstack([baseline_data, post_data])
n_base = baseline_data.shape[0]
rng = np.random.RandomState(RANDOM_SEED)
N_PERM = 1000
null_cwj = np.zeros(N_PERM)

for p in range(N_PERM):
    pidx = rng.permutation(all_data.shape[0])
    pdata = all_data[pidx, :]
    ca = compute_spearman_matrix(pdata[:n_base, :])
    cb = compute_spearman_matrix(pdata[n_base:, :])
    null_cwj[p] = continuous_wj(ca, cb)
    if (p+1) % 200 == 0:
        print(f"  Permutation {p+1}/{N_PERM}")

perm_p = np.mean(null_cwj >= obs_cwj)
z_score = (obs_cwj - null_cwj.mean()) / null_cwj.std()

print(f"  Observed continuous WJ (post vs baseline): {obs_cwj:.6f}")
print(f"  Permutation p-value: {perm_p:.6f}")
print(f"  z-score: {z_score:.2f}")
print(f"  Control continuous WJ (pre vs baseline): {obs_cwj_control:.6f}")

# Save
df = pd.DataFrame({
    'window_center': [pd.Timestamp(t).strftime('%Y-%m-%d') for t in wj_times],
    'continuous_wj': cwj_vals,
    'binary_j': bwj_vals,
})
df.to_csv(os.path.join(RESULTS_DIR, 'climate_continuous_wj_trajectory_20260314.csv'),
          index=False, float_format='%.6f')

summary = {
    'continuous_wj_tau': float(tau_cwj),
    'continuous_wj_p': float(p_cwj),
    'binary_j_tau': float(tau_bwj),
    'binary_j_p': float(p_bwj),
    'metric_correlation_rho': float(corr_metrics),
    'obs_continuous_wj_post': float(obs_cwj),
    'perm_p_continuous': float(perm_p),
    'z_score_continuous': float(z_score),
    'obs_continuous_wj_control': float(obs_cwj_control),
}
with open(os.path.join(RESULTS_DIR, 'continuous_wj_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to {RESULTS_DIR}")
print("=" * 60)
