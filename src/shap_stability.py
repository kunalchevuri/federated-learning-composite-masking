"""
SHAP attribution stability across the 10 evaluation seeds.

For each feature set and each of the 10 evaluation seeds, loads the
validation-selected centralized model saved by revision_experiments.py,
explains 300 test rows with shap.GradientExplainer (100 training rows as
background), and reports per dimension:
  * group importance: SHAP values are additive, so a feature group's
    contribution for one person is the sum of its columns' signed SHAP values;
    importance is the mean absolute group contribution (one-hot blocks are not
    inflated by summing absolute values of their columns)
  * a 95% bootstrap CI of that importance over seeds
  * how often each group is the top-ranked group across seeds
  * Kendall's W: agreement of the full group ranking across seeds (1 = identical)
  * direction: correlation between a single-column feature's value and its
    SHAP value (positive = higher value, higher predicted risk)

Outputs: results/revision_shap_by_seed.csv, results/revision_shap_summary.csv
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revision_experiments as rx  # noqa: E402

DIMS = rx.DIM_NAMES
N_EXPLAIN, N_BACKGROUND = 300, 100
N_WORKERS = int(os.environ.get("SHAP_WORKERS", 10))


def design_columns(fs):
    """Column names in rx.design_matrix order (same construction, names kept)."""
    cols = ['age', 'female', 'pir', 'sys', 'dia'] + [f'race_{r}' for r in (1, 2, 3, 4, 6, 7)]
    if fs == 'rich':
        cols += ['education', 'household_size', 'us_born', 'pulse', 'hba1c', 'total_chol', 'hdl',
                 'insured', 'routine_care', 'care_visits', 'hospitalized', 'diabetes_dx', 'htn_dx',
                 'high_chol_dx', 'asthma_dx', 'cvd_dx', 'cancer_dx', 'n_prescriptions',
                 'marital_married', 'marital_never', 'marital_prev_married',
                 'smoke_current', 'smoke_former', 'smoke_never',
                 'lab_missing', 'mcq_missing', 'edu_missing']
    return cols


def group_of(c):
    if c.startswith('race_'):
        return 'race/ethnicity'
    if c.startswith('marital_'):
        return 'marital status'
    if c.startswith('smoke_'):
        return 'smoking'
    if c.endswith('_missing'):
        return 'missing-data indicators'
    return {'pir': 'income ratio', 'female': 'sex (female)', 'sys': 'systolic BP', 'dia': 'diastolic BP',
            'total_chol': 'cholesterol (total, HDL)', 'hdl': 'cholesterol (total, HDL)'}.get(c, c)


def explain(job):
    import shap
    import torch
    torch.set_num_threads(1)
    fs, pt = job
    seed = int(re.search(r'__s(\d+)__', pt).group(1))
    rng = np.random.default_rng(1000 + seed)
    df = rx.load_data()
    idx_tr, _, idx_te = rx.split(df, seed, True)
    X = rx.design_matrix(df, fs, idx_tr)
    cols = design_columns(fs)
    assert X.shape[1] == len(cols), (fs, X.shape, len(cols))
    groups = np.array([group_of(c) for c in cols])
    model = rx.MultiOutputMLP(X.shape[1])
    model.load_state_dict(torch.load(pt))
    model.eval()
    bg = torch.tensor(X[rng.choice(idx_tr, N_BACKGROUND, replace=False)])
    ex = torch.tensor(X[rng.choice(idx_te, N_EXPLAIN, replace=False)])
    sv = shap.GradientExplainer(model, bg).shap_values(ex)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)      # (n, f, 7)
    xe = ex.numpy()
    rows = []
    for d, dname in enumerate(DIMS):
        for g in dict.fromkeys(groups):
            idx = np.where(groups == g)[0]
            contrib = sv[:, idx, d].sum(axis=1)                                   # per-person group contribution
            rho = (float(np.corrcoef(xe[:, idx[0]], sv[:, idx[0], d])[0, 1])
                   if len(idx) == 1 and xe[:, idx[0]].std() > 0 else np.nan)
            rows.append({'fs': fs, 'seed': seed, 'dimension': dname, 'group': g,
                         'mean_abs_shap': float(np.abs(contrib).mean()), 'direction_r': rho})
    return rows


def kendall_w(ranks):
    """ranks: (m raters, n items)."""
    m, n = ranks.shape
    R = ranks.sum(0)
    S = ((R - R.mean()) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))


def main():
    jobs = [(fs, pt) for fs in ['orig6', 'rich']
            for pt in sorted(glob.glob(f"{rx.OUT_DIR}eval__{fs}__centralized__s*__lr*.pt"))]
    with Pool(N_WORKERS) as pool:
        rows = [r for out in pool.map(explain, jobs) for r in out]
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv("results/revision_shap_by_seed.csv", index=False)

    rng = np.random.default_rng(11)
    out = []
    for (fs, dname), g in by_seed.groupby(['fs', 'dimension']):
        piv = g.pivot(index='seed', columns='group', values='mean_abs_shap')
        ranks = np.vstack([rankdata(-r) for r in piv.values])
        top = piv.idxmax(axis=1).value_counts()
        W = kendall_w(ranks)
        for grp in piv.columns:
            v = piv[grp].values
            b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
            dr = g[g['group'] == grp]['direction_r']
            out.append({'fs': fs, 'dimension': dname, 'group': grp, 'n_seeds': len(v),
                        'mean_abs_shap': v.mean(), 'ci_lo': np.percentile(b, 2.5),
                        'ci_hi': np.percentile(b, 97.5), 'top_share': top.get(grp, 0) / len(v),
                        'kendall_w': W, 'direction_r': dr.mean(),
                        'direction_consistent': bool((np.sign(dr) == np.sign(dr.mean())).all())
                        if dr.notna().all() else np.nan})
    summ = pd.DataFrame(out)
    summ.to_csv("results/revision_shap_summary.csv", index=False)
    pd.set_option('display.width', 200)
    top = summ.sort_values('mean_abs_shap', ascending=False).groupby(['fs', 'dimension']).head(3)
    print(top.round(3).sort_values(['fs', 'dimension', 'mean_abs_shap'], ascending=[True, True, False])
          .to_string(index=False))


if __name__ == '__main__':
    main()
