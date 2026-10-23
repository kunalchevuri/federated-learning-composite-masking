"""
Null calibration of the Masking Gap (MG) for the nine baseline-stage conditions.

MG = mean_d AUC_d - min_d AUC_d is non-negative by construction: even a model
that outputs random scores has MG > 0 because each per-client AUC is estimated
on a small test fold. This script rebuilds each seed's exact test fold
(train_test_split(test_size=0.2, random_state=seed, stratify=client_id), as in
revision_experiments.py), derives the random-score null distribution of MG from the
exact positive/negative counts, and asks whether observed MG exceeds it.

Input: results/baseline_results.csv (baseline_to_csv.py). Cells with fewer than 5
positives or 5 negatives are written there as 0.5; the primary analysis ("evaluable")
treats them as missing. "as_reported" keeps the 0.5 values (sensitivity check).

Null model: for a cell with k positives among n, AUC under random scores is
(rank-sum of positives - k(k+1)/2) / (k(n-k)) with positives occupying a
uniformly random k-subset of ranks. Dimensions are drawn independently.

Outputs (results/):
  null_calibration_clients.csv   per variant x method x seed x client: MG, null quantiles, MG-excess, p, tier
  null_calibration_summary.csv   per variant x method: MG, MG-excess + 95% cluster-bootstrap CI, Wilcoxon vs 0
  null_calibration_per_seed.csv  per method x seed: reported MG, evaluable MG, MG-excess
  evaluability.csv               per seed x dimension: clients with an evaluable AUC
  population_mg.csv              per method: population-level MG vs its own null
  population_dim_auc.csv         per method x dimension: client-averaged AUC + 95% CI
  pairwise_tests.csv             prespecified paired comparisons
  delta_auc_decoupling.csv       per-dimension AUC change from decoupling
  argmin_vs_null.csv             which dimension is the minimum, observed vs null
  mixed_model.txt                MG ~ method + seed + (1 | client)
  null_power_curve.csv           null MG median / 95th / 99th percentile vs test-set size

Figures (figures/):
  fig11_null_calibration.png     (a) MG-excess by method, (b) null-percentile histogram
  fig12_power_curve.png          null MG vs per-client test-set size
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi']        = 150

# ── paths ──────────────────────────────────────────────────────────────────────
results_path = "results/baseline_results.csv"
nhanes_path  = "results/nhanes_clean.csv"
results_dir  = "results/"
figures_dir  = "figures/"
os.makedirs(figures_dir, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
DIMS       = ['Sleep', 'Activity', 'Sedentary', 'Mental', 'Metabolic', 'Wellbeing', 'Diet']
DIM_COLS   = ['auc_sleep', 'auc_activity', 'auc_sedentary', 'auc_mental',
              'auc_metabolic', 'auc_wellbeing', 'auc_diet']
LABEL_COLS = ['sleep_poor', 'activity_poor', 'sedentary_poor', 'mental_poor',
              'metabolic_poor', 'wellbeing_poor', 'diet_poor']
SEEDS      = [42, 123, 456]
MIN_CLASS  = 5          # evaluability rule: >= 5 positives and >= 5 negatives

# (method, condition label) in paper order; 'centralized_weighted' is the
# CSV name of C2 (seven independent single-output classifiers)
METHODS = [
    ('centralized',          'C1 Centralized'),
    ('centralized_weighted', 'C2 Sep. classifiers'),
    ('federated_separate',   'C2-FL Fed. separate'),
    ('fedavg',               'C3 FedAvg'),
    ('scaffold',             'C4 SCAFFOLD'),
    ('csagg',                'C5 CS-Agg'),
    ('ditto',                'C6 Ditto'),
    ('qfedavg',              'C7 q-FedAvg'),
    ('seqlocal_fedavg',      'C8 Per-dim.'),
]
METHOD_LABEL = dict(METHODS)

# prespecified paired comparisons (a, b): tests MG(a) - MG(b)
PAIRS = [
    ('centralized_weighted', 'centralized'),
    ('federated_separate',   'fedavg'),
    ('seqlocal_fedavg',      'fedavg'),
    ('qfedavg',              'fedavg'),
    ('ditto',                'fedavg'),
    ('fedavg',               'centralized'),
]

N_NULL = 10000
N_BOOT = 10000
POWER_N = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]

INK      = '#1F2933'
NULL_COL = '#8C959F'   # null distribution: recessive gray
OBS_COL  = '#2A6FB0'   # observed: single accent

rng = np.random.default_rng(42)

# ── load data ──────────────────────────────────────────────────────────────────
df     = pd.read_csv(results_path)
nhanes = pd.read_csv(nhanes_path)

# final round per (method, seed); centralized conditions have a single round 0
last = df[df['round'] == df.groupby(['method', 'seed'])['round'].transform('max')].copy()
last = last.sort_values(['method', 'seed', 'client_id']).reset_index(drop=True)
print(f"Loaded: baseline_results {df.shape} | final-round rows {last.shape}")

# exact per-seed test folds, rebuilt with the framework's split
nhanes['client_id'] = (
    pd.cut(nhanes['INDFMPIR'], [0, 1.0, 2.0, 3.0, 4.0, 99], labels=False, right=False) * 4
    + pd.cut(nhanes['RIDAGEYR'], [18, 36, 51, 65, 120], labels=False, right=False))
n_pos, n_tot = {}, {}
for seed in SEEDS:
    _, idx_te = train_test_split(nhanes.index, test_size=0.2, random_state=seed,
                                 stratify=nhanes['client_id'])
    te = nhanes.loc[idx_te]
    n_pos[seed] = te.groupby('client_id')[LABEL_COLS].sum().astype(int).values   # (20, 7)
    n_tot[seed] = te.groupby('client_id').size().values                          # (20,)
    csv_n = last[(last['method'] == 'fedavg') & (last['seed'] == seed)]['n_test'].values
    assert (csv_n == n_tot[seed]).all(), f"seed {seed}: rebuilt test fold does not match CSV"
print("  rebuilt test folds match n_test in the CSV for all seeds")

K = np.stack([n_pos[s][c] for s, c in zip(last['seed'], last['client_id'])])      # (rows, 7)
N = np.array([n_tot[s][c] for s, c in zip(last['seed'], last['client_id'])])
evaluable = (K >= MIN_CLASS) & ((N[:, None] - K) >= MIN_CLASS)

ev_rows = []
for seed in SEEDS:
    e = (n_pos[seed] >= MIN_CLASS) & ((n_tot[seed][:, None] - n_pos[seed]) >= MIN_CLASS)
    ev_rows.append({'seed': seed, **{d: int(e[:, i].sum()) for i, d in enumerate(DIMS)}})
evaluability = pd.DataFrame(ev_rows)
evaluability.to_csv(os.path.join(results_dir, 'evaluability.csv'), index=False)
print("  clients (of 20) with an evaluable AUC:")
print(evaluability.to_string(index=False))

# a cell's reported value is a 0.5 placeholder unless it is a centralized
# seed-42/123 cell with at least one positive and one negative
legacy = np.zeros(len(last), bool)   # no low-count AUCs in the baseline runs
placeholder = ~evaluable & ~(legacy[:, None] & (K >= 1) & ((N[:, None] - K) >= 1))
assert np.allclose(last[DIM_COLS].values[placeholder], 0.5), "placeholder cells are not 0.5"
print(f"  placeholder cells: {int(placeholder.sum())} of {placeholder.size}; "
      f"legacy low-count centralized cells: {int((~evaluable & ~placeholder).sum())}")


# ── helpers ────────────────────────────────────────────────────────────────────
def null_auc(k, n, size):
    """AUC of random scores with k positives among n samples, `size` draws."""
    ranks = rng.random((size, n)).argsort(axis=1)[:, :k] + 1
    return (ranks.sum(axis=1) - k * (k + 1) / 2) / (k * (n - k))


def masking_gap(aucs, axis=0):
    return np.nanmean(aucs, axis=axis) - np.nanmin(aucs, axis=axis)


def cluster_boot_ci(values, n_boot=N_BOOT):
    """95% CI of the mean, resampling clients (values are one per client)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def hodges_lehmann(diff):
    """Median of Walsh averages - the paired effect size matching Wilcoxon."""
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def savefig(path):
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  -> {os.path.basename(path)}  ({os.path.getsize(path)/1024:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 1. NULL AUC DRAWS PER (seed, client, dimension)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Null AUC draws from exact test-fold counts")
null_draw = {}   # (seed, client, dim) -> (N_NULL,)
for seed in SEEDS:
    for c in range(20):
        for d in range(7):
            k, n = int(n_pos[seed][c, d]), int(n_tot[seed][c])
            if 1 <= k <= n - 1:
                null_draw[(seed, c, d)] = null_auc(k, n, N_NULL)
CONST_HALF = np.full(N_NULL, 0.5)


def null_matrix(seed, c, dims_used, variant):
    """(len(dims_used), N_NULL) null AUCs; placeholders are constant 0.5."""
    rows = []
    for d in dims_used:
        k, n = int(n_pos[seed][c, d]), int(n_tot[seed][c])
        use_draw = (k >= MIN_CLASS and n - k >= MIN_CLASS) or (variant == 'legacy' and 1 <= k <= n - 1)
        rows.append(null_draw[(seed, c, d)] if use_draw else CONST_HALF)
    return np.stack(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLIENT-LEVEL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Client-level MG vs null")


def calibrate_clients(variant):
    """variant 'evaluable': only cells with >=5/>=5; 'as_reported': the paper's values."""
    aucs = last[DIM_COLS].values.astype(float)
    if variant == 'evaluable':
        aucs = np.where(evaluable, aucs, np.nan)
    out = last[['method', 'seed', 'client_id', 'n_test']].copy()
    out['n_dims'] = (~np.isnan(aucs)).sum(axis=1)
    out['composite_auc'] = np.nanmean(aucs, axis=1)
    out['masking_gap']   = masking_gap(aucs, axis=1)
    out['argmin_dim']    = [DIMS[int(np.nanargmin(a))] for a in aucs]
    cache, stats = {}, []
    for i, (m, seed, c, mg) in enumerate(zip(out['method'], out['seed'], out['client_id'],
                                             out['masking_gap'])):
        dims_used = tuple(np.where(~np.isnan(aucs[i]))[0])
        leg = variant == 'as_reported' and legacy[i]
        key = (seed, c, dims_used, leg)
        if key not in cache:
            cache[key] = masking_gap(null_matrix(seed, c, dims_used, 'legacy' if leg else 'std'))
        nd = cache[key]
        stats.append((nd.mean(), np.percentile(nd, 95), np.percentile(nd, 99),
                      (nd < mg).mean(), ((nd >= mg).sum() + 1) / (N_NULL + 1)))
    out[['mg_null_mean', 'mg_null_p95', 'mg_null_p99', 'null_pctile', 'p_value']] = stats
    out['mg_excess'] = out['masking_gap'] - out['mg_null_mean']
    out['tier'] = np.select(
        [out['masking_gap'] >= out['mg_null_p99'], out['masking_gap'] >= out['mg_null_p95']],
        ['severe (>=null p99)', 'elevated (>=null p95)'],
        default='within noise')
    out['variant'] = variant
    return out


def summarize(cl):
    rows = []
    for method, label in METHODS:
        g = cl[cl['method'] == method]
        per_client = g.groupby('client_id')['mg_excess'].mean()   # seed-averaged, n=20
        lo, hi = cluster_boot_ci(per_client)
        _, p = wilcoxon(per_client, alternative='two-sided')
        rows.append({
            'variant': g['variant'].iloc[0], 'method': method, 'label': label,
            'composite_auc': g['composite_auc'].mean(),
            'mg_obs': g['masking_gap'].mean(),
            'mg_null': g['mg_null_mean'].mean(),
            'mg_excess': per_client.mean(), 'ci_lo': lo, 'ci_hi': hi,
            'wilcoxon_p': p,
            'frac_ge_null_p95': (g['masking_gap'] >= g['mg_null_p95']).mean(),
            'frac_ge_null_p99': (g['masking_gap'] >= g['mg_null_p99']).mean(),
            'mean_null_pctile': g['null_pctile'].mean(),
        })
    s = pd.DataFrame(rows)
    s['wilcoxon_p_holm'] = multipletests(s['wilcoxon_p'], method='holm')[1]
    return s


cl          = calibrate_clients('evaluable')
cl_reported = calibrate_clients('as_reported')
summary          = summarize(cl)
summary_reported = summarize(cl_reported)
pd.concat([cl, cl_reported]).to_csv(
    os.path.join(results_dir, 'null_calibration_clients.csv'), index=False)
pd.concat([summary, summary_reported]).to_csv(
    os.path.join(results_dir, 'null_calibration_summary.csv'), index=False)

show = ['label', 'mg_obs', 'mg_null', 'mg_excess', 'ci_lo', 'ci_hi', 'wilcoxon_p_holm',
        'frac_ge_null_p95', 'frac_ge_null_p99', 'mean_null_pctile']
print("  Evaluable cells only (primary):")
print(summary[show].round(3).to_string(index=False))
print("\n  As reported in the submission (0.5 placeholders included):")
print(summary_reported[show].round(3).to_string(index=False))
print("  (under the null: frac_ge_null_p95 ~ 0.05, frac_ge_null_p99 ~ 0.01, mean_null_pctile ~ 0.50)")

print("\n  Tier counts (client x seed, evaluable):")
print(pd.crosstab(cl['method'].map(METHOD_LABEL), cl['tier']).reindex(
    [l for _, l in METHODS]).to_string())

per_seed = pd.concat({
    'mg_reported':  cl_reported.groupby(['method', 'seed'])['masking_gap'].mean(),
    'mg_evaluable': cl.groupby(['method', 'seed'])['masking_gap'].mean(),
    'mg_null':      cl.groupby(['method', 'seed'])['mg_null_mean'].mean(),
    'mg_excess':    cl.groupby(['method', 'seed'])['mg_excess'].mean(),
}, axis=1).unstack('seed').round(4)
per_seed.index = per_seed.index.map(METHOD_LABEL)
per_seed = per_seed.reindex([l for _, l in METHODS])
per_seed.to_csv(os.path.join(results_dir, 'null_calibration_per_seed.csv'))
print("\n  Per-seed MG (reported / evaluable) and MG-excess:")
print(per_seed.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. POPULATION-LEVEL MG (client-averaged AUC per dimension, evaluable cells)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Population-level MG vs null")


def pop_null(seed):
    """Null of per-seed population MG: average null AUC over evaluable clients."""
    e = (n_pos[seed] >= MIN_CLASS) & ((n_tot[seed][:, None] - n_pos[seed]) >= MIN_CLASS)
    dim_means = np.stack([np.mean([null_draw[(seed, c, d)] for c in range(20) if e[c, d]], axis=0)
                          for d in range(7)])
    return masking_gap(dim_means)


null_pop_mg = np.mean([pop_null(s) for s in SEEDS], axis=0)   # mean of per-seed nulls
print(f"  null population MG: mean {null_pop_mg.mean():.3f}  95th {np.percentile(null_pop_mg, 95):.3f}")

ev_auc = last[['method', 'seed', 'client_id']].join(
    pd.DataFrame(np.where(evaluable, last[DIM_COLS].values, np.nan), columns=DIM_COLS))
pop_rows, dim_rows = [], []
for method, label in METHODS:
    g = ev_auc[ev_auc['method'] == method]
    seed_mg = [masking_gap(s[DIM_COLS].mean().values) for _, s in g.groupby('seed')]
    obs = float(np.mean(seed_mg))
    pc = g.groupby('client_id')[DIM_COLS].mean()   # seed-averaged, NaN-aware
    idx = rng.integers(0, len(pc), size=(N_BOOT, len(pc)))
    boot_mg = [masking_gap(np.nanmean(pc.values[i], axis=0)) for i in idx]
    pop_rows.append({
        'method': method, 'label': label, 'pop_mg': obs,
        'boot_ci_lo': np.percentile(boot_mg, 2.5), 'boot_ci_hi': np.percentile(boot_mg, 97.5),
        'null_mean': null_pop_mg.mean(), 'null_p95': np.percentile(null_pop_mg, 95),
        'p_value': ((null_pop_mg >= obs).sum() + 1) / (N_NULL + 1),
        'worst_dim': DIMS[int(np.nanargmin(pc.mean().values))],
    })
    for d, col in zip(DIMS, DIM_COLS):
        lo, hi = cluster_boot_ci(pc[col])
        dim_rows.append({'method': method, 'label': label, 'dimension': d,
                         'n_clients': int(pc[col].notna().sum()),
                         'auc_mean': pc[col].mean(), 'ci_lo': lo, 'ci_hi': hi,
                         'above_chance': lo > 0.5})
pop = pd.DataFrame(pop_rows)
pop['p_holm'] = multipletests(pop['p_value'], method='holm')[1]
pop.to_csv(os.path.join(results_dir, 'population_mg.csv'), index=False)
dim_df = pd.DataFrame(dim_rows)
dim_df.to_csv(os.path.join(results_dir, 'population_dim_auc.csv'), index=False)
print(pop[['label', 'pop_mg', 'boot_ci_lo', 'boot_ci_hi', 'null_p95', 'p_value', 'p_holm', 'worst_dim']]
      .round(4).to_string(index=False))

print("\n  Client-averaged AUC per dimension (* = 95% CI above 0.5):")
tab = dim_df.assign(cell=lambda t: t['auc_mean'].map('{:.3f}'.format)
                    + np.where(t['above_chance'], '*', ' '))
print(tab.pivot(index='label', columns='dimension', values='cell')
      .reindex([l for _, l in METHODS])[DIMS].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. PAIRED COMPARISONS + MIXED MODEL (evaluable MG)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Paired comparisons (seed-averaged per client, n=20, two-sided, Holm)")
client_mg = cl.groupby(['method', 'client_id'])['masking_gap'].mean().unstack('method')
pair_rows = []
for a, b in PAIRS:
    diff = (client_mg[a] - client_mg[b]).dropna()
    _, p = wilcoxon(diff, alternative='two-sided')
    idx = rng.integers(0, len(diff), size=(2000, len(diff)))
    hl_boot = [hodges_lehmann(diff.values[i]) for i in idx]
    pair_rows.append({
        'comparison': f"{METHOD_LABEL[a]} - {METHOD_LABEL[b]}",
        'mean_diff': diff.mean(), 'hodges_lehmann': hodges_lehmann(diff),
        'hl_ci_lo': np.percentile(hl_boot, 2.5), 'hl_ci_hi': np.percentile(hl_boot, 97.5),
        'n_clients_lower': int((diff < 0).sum()), 'n_clients': len(diff), 'wilcoxon_p': p,
    })
pairs = pd.DataFrame(pair_rows)
pairs['wilcoxon_p_holm'] = multipletests(pairs['wilcoxon_p'], method='holm')[1]
pairs.to_csv(os.path.join(results_dir, 'pairwise_tests.csv'), index=False)
print(pairs.round(4).to_string(index=False))

delta_rows = []
for a, b in [('federated_separate', 'fedavg'), ('centralized_weighted', 'centralized'),
             ('seqlocal_fedavg', 'fedavg')]:
    ca = ev_auc[ev_auc['method'] == a].groupby('client_id')[DIM_COLS].mean()
    cb = ev_auc[ev_auc['method'] == b].groupby('client_id')[DIM_COLS].mean()
    for d, col in zip(DIMS, DIM_COLS):
        diff = (ca[col] - cb[col]).dropna()
        lo, hi = cluster_boot_ci(diff)
        delta_rows.append({'comparison': f"{METHOD_LABEL[a]} - {METHOD_LABEL[b]}",
                           'dimension': d, 'delta_auc': diff.mean(), 'ci_lo': lo, 'ci_hi': hi,
                           'ci_excludes_0': (lo > 0) or (hi < 0)})
delta = pd.DataFrame(delta_rows)
delta.to_csv(os.path.join(results_dir, 'delta_auc_decoupling.csv'), index=False)
print("\n  Per-dimension AUC change (95% cluster-bootstrap CI):")
print(delta.round(3).to_string(index=False))

mm = smf.mixedlm("masking_gap ~ C(method, Treatment('fedavg')) + C(seed)",
                 data=cl, groups=cl['client_id']).fit(reml=True)
with open(os.path.join(results_dir, 'mixed_model.txt'), 'w') as fh:
    fh.write(str(mm.summary()))
print("\n  Mixed model MG ~ method + seed + (1|client), reference = FedAvg:")
coefs = mm.summary().tables[1]
print(coefs[coefs.index.str.contains('method')].to_string())
re_var, resid_var = float(mm.cov_re.iloc[0, 0]), float(mm.scale)
print(f"  variance: client {re_var:.5f}  residual {resid_var:.5f}  "
      f"(client share {re_var / (re_var + resid_var):.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. WHICH DIMENSION IS THE MINIMUM? OBSERVED VS NULL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Identity of the worst dimension, observed vs random-score null")
null_argmin = np.zeros(7)
n_cells = 0
for seed in SEEDS:
    for c in range(20):
        dims_used = [d for d in range(7)
                     if n_pos[seed][c, d] >= MIN_CLASS and n_tot[seed][c] - n_pos[seed][c, d] >= MIN_CLASS]
        am = np.argmin(np.stack([null_draw[(seed, c, d)] for d in dims_used]), axis=0)
        null_argmin += np.bincount(np.array(dims_used)[am], minlength=7) / N_NULL
        n_cells += 1
null_share = null_argmin / n_cells
obs_share = cl['argmin_dim'].value_counts(normalize=True).reindex(DIMS).fillna(0).values
prevalence = nhanes[LABEL_COLS].mean().values
argmin_tab = pd.DataFrame({'dimension': DIMS, 'prevalence': prevalence,
                           'observed_share': obs_share, 'null_share': null_share})
argmin_tab.to_csv(os.path.join(results_dir, 'argmin_vs_null.csv'), index=False)
print(argmin_tab.round(3).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6. POWER CURVE: NULL MG VS TEST-SET SIZE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Null MG vs per-client test-set size")
power_rows = []
for n in POWER_N:
    size = 4000 if n <= 1000 else 1000
    aucs = np.stack([null_auc(int(np.clip(round(p * n), 1, n - 1)), n, size) for p in prevalence])
    mg = masking_gap(aucs)
    power_rows.append({'n_test': n, 'null_median': np.median(mg),
                       'null_p95': np.percentile(mg, 95), 'null_p99': np.percentile(mg, 99)})
power = pd.DataFrame(power_rows)
power.to_csv(os.path.join(results_dir, 'null_power_curve.csv'), index=False)
print(power.round(4).to_string(index=False))
for thr in (0.10, 0.05, 0.02):
    ok = power[power['null_p95'] < thr]
    need = int(ok['n_test'].iloc[0]) if len(ok) else None
    print(f"  smallest n_test (grid) with null 95th < {thr:.2f}: {need}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — NULL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 11] Null calibration")
# single-column figure (IEEE column = 3.5 in); condition names are in Table II
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.5, 2.35),
                                 gridspec_kw={'width_ratios': [1, 1.05]})

s = summary.iloc[::-1].reset_index(drop=True)
y = np.arange(len(s))
ax_a.axvline(0, color=NULL_COL, lw=0.8, ls='--', zorder=1)
ax_a.hlines(y, s['ci_lo'], s['ci_hi'], color=OBS_COL, lw=1.5, zorder=2)
ax_a.plot(s['mg_excess'], y, 'o', color=OBS_COL, ms=4, mec='white', mew=0.6, zorder=3)
ax_a.set_yticks(y)
ax_a.set_yticklabels(s['label'].str.split().str[0], fontsize=7, color=INK)
ax_a.set_xlabel('MG-excess', fontsize=7.5)
ax_a.set_title('(a) MG minus chance', fontsize=7.5, loc='left', color=INK)
ax_a.set_xticks([-0.01, 0, 0.01, 0.02])
ax_a.set_xticklabels(['\u22120.01', '0', '0.01', '0.02'])
ax_a.tick_params(axis='x', labelsize=6.5)

bins = np.linspace(0, 1, 11)
ax_b.hist(cl['null_pctile'], bins=bins, color=OBS_COL, edgecolor='white', linewidth=1)
expected = len(cl) / (len(bins) - 1)
ax_b.axhline(expected, color=NULL_COL, lw=1, ls='--')
ax_b.set_xlabel('Percentile within null', fontsize=7.5)
ax_b.set_ylabel('Results', fontsize=7.5)
ax_b.set_title('(b) Null percentile', fontsize=7.5, loc='left', color=INK)
ax_b.set_ylim(0, expected * 1.5)
ax_b.set_xticks([0, 0.5, 1])
ax_b.set_xticklabels(['0', '0.5', '1'])
ax_b.tick_params(labelsize=6.5)

plt.tight_layout(w_pad=0.8)
savefig(os.path.join(figures_dir, 'fig11_null_calibration.png'))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — POWER CURVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 12] Null MG vs test-set size")
# drawn at the size it is printed (0.82 of a 3.5 in column)
fig, ax = plt.subplots(figsize=(2.87, 2.2))
ax.axvspan(N.min(), N.max(), color=NULL_COL, alpha=0.15, lw=0)
ax.text(np.sqrt(N.min() * N.max()), 0.005, 'this study\n(per client)',
        fontsize=6.5, color=INK, ha='center', va='bottom')
ax.plot(power['n_test'], power['null_p95'], color=OBS_COL, lw=1.5, label='null 95th percentile')
ax.plot(power['n_test'], power['null_median'], color=NULL_COL, lw=1.5, label='null median')
obs_mean = cl['masking_gap'].mean()
ax.axhline(obs_mean, color=INK, lw=0.8, ls=':', label=f'observed mean MG ({obs_mean:.3f})')
ax.legend(fontsize=6.5, frameon=False, loc='upper right', handlelength=1.8)
ax.set_xscale('log')
ax.set_xlabel('Per-client test-set size', fontsize=7.5)
ax.set_ylabel('MG of random scores', fontsize=7.5)
ax.set_ylim(0, max(power['null_p95'].max(), obs_mean) * 1.1)
ax.tick_params(labelsize=6.5)
plt.tight_layout()
savefig(os.path.join(figures_dir, 'fig12_power_curve.png'))

print("\nDone.")
