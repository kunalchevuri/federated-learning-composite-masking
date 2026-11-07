"""
Robustness checks on the Masking Gap, the composite blind spot and the seed-level contrasts.

  equal-AUC null   The random-score and label-permutation nulls assume scores carry no
                   information. Here the reference is a model with real but EQUAL skill on
                   every dimension: per client and dimension, scores are binormal (positives
                   N(delta, 1), negatives N(0, 1)) with delta = sqrt(2) Phi^-1(A), where A is
                   the run's pooled macro-AUC, drawn at the cell's exact class counts.
                   MG above this null means the dimensions truly differ in AUC.
  composite weights Delta_d for composites other than the equal-weight mean: weights
                   proportional to prevalence, and 1,000 random Dirichlet(1) weight vectors.
  corrected t-test Nadeau-Bengio corrected resampled t-test for the seed-level contrasts:
                   the test sets of different seeds overlap, so the variance of the mean
                   difference is inflated by (1/J + n_test/n_train) instead of 1/J.

Inputs: results/revision/{eval,baseline}__*.npz, results/revision_runs.csv
Outputs: results/robust_equal_auc_null.csv, results/robust_composite_weights.csv,
         results/robust_corrected_ttest.csv
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as tdist, wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_revision as ar  # noqa: E402

REV = "results/revision/"
N_NULL = 1000
N_WEIGHTS = 1000
N_BOOT = 2000
TEST_TRAIN = 0.2 / 0.7     # test fold over the training data actually fitted (validation held out)
rng = np.random.default_rng(11)


def run_paths():
    return sorted(glob.glob(REV + 'eval__*.npz') + glob.glob(REV + 'baseline__*.npz'))


def parse(path):
    m = re.match(r'(eval|baseline)__(\w+?)__([\w.]+)__s(\d+)__lr', os.path.basename(path))
    stage, fs, cond, seed = m.groups()
    cond = re.sub(r'^(fedprox)_mu[\d.]+$|^(ditto)_lam[\d.]+$|^(qfedavg)_q[\d.]+$|^(csagg)_w\d+$',
                  lambda mm: next(g for g in mm.groups() if g), cond)
    return stage, fs, {'scaffold_adam': 'scaffold'}.get(cond, cond), int(seed)


def equal_auc_null_mg(ks, n, delta):
    """N_NULL draws of MG when every dimension has the same binormal AUC, at counts ks."""
    aucs = []
    for k in ks:
        s = rng.standard_normal((N_NULL, n))
        s[:, :k] += delta
        R = rankdata(s, axis=1)
        aucs.append((R[:, :k].sum(1) - k * (k + 1) / 2) / (k * (n - k)))
    a = np.array(aucs).T
    return a.mean(1) - a.min(1)


def equal_auc_null():
    rows = []
    for path in run_paths():
        stage, fs, cond, seed = parse(path)
        z = np.load(path)
        y, p, cid = z['y_test'], z['p_test_best'], z['client_test']
        ok = ar.evaluable(y)
        A = np.mean([roc_auc_score(y[:, d], p[:, d]) for d in np.where(ok)[0]])
        delta = np.sqrt(2) * norm.ppf(A)
        for c in np.unique(cid):
            m = cid == c
            okc = ar.evaluable(y[m])
            _, mg, _ = ar.perm_null_mg(y[m], p[m], okc, n_perm=1)
            null = equal_auc_null_mg(y[m][:, okc].sum(0).astype(int), int(m.sum()), delta)
            rows.append({'stage': stage, 'fs': fs, 'cond': cond, 'seed': seed, 'client_id': int(c),
                         'pooled_macro_auc': A, 'mg': mg, 'eq_null_mean': null.mean(),
                         'eq_null_p95': np.percentile(null, 95)})
    cl = pd.DataFrame(rows)
    cl['eq_excess'] = cl['mg'] - cl['eq_null_mean']
    cl['above_p95'] = cl['mg'] > cl['eq_null_p95']
    # clients as units: average over seeds, then cluster bootstrap and Wilcoxon over clients
    out = []
    for (stage, fs, cond), g in cl.groupby(['stage', 'fs', 'cond']):
        v = g.groupby('client_id')['eq_excess'].mean().values
        b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        out.append({'stage': stage, 'fs': fs, 'cond': cond, 'mg': g['mg'].mean(),
                    'eq_null': g['eq_null_mean'].mean(), 'eq_excess': v.mean(),
                    'ci_lo': np.percentile(b, 2.5), 'ci_hi': np.percentile(b, 97.5),
                    'share_above_p95': g['above_p95'].mean(), 'p': wilcoxon(v).pvalue})
    out = pd.DataFrame(out)
    out['p_holm'] = multipletests(out['p'], method='holm')[1]
    out.to_csv('results/robust_equal_auc_null.csv', index=False)
    return out


def composite_weights():
    prev = pd.read_csv(ar.NHANES_PATH)[ar.LABEL_COLS].mean().values
    W = rng.dirichlet(np.ones(7), N_WEIGHTS)
    rows = []
    for path in glob.glob(REV + 'eval__*.npz'):
        stage, fs, cond, seed = parse(path)
        z = np.load(path)
        y, p = z['y_test'], z['p_test_best']
        comps = {'equal': p.mean(1), 'prevalence': p @ (prev / prev.sum())}
        for d in range(7):
            own = roc_auc_score(y[:, d], p[:, d])
            r = {'fs': fs, 'cond': cond, 'seed': seed, 'dimension': ar.DIMS[d]}
            for name, s in comps.items():
                r[f'delta_{name}'] = own - roc_auc_score(y[:, d], s)
            # rank-based AUC for all random composites at once
            Rr = rankdata(p @ W.T, axis=0)
            k, n = y[:, d].sum(), len(y)
            a = (Rr[y[:, d] == 1].sum(0) - k * (k + 1) / 2) / (k * (n - k))
            dr = own - a
            r.update(delta_random_median=np.median(dr), delta_random_p05=np.percentile(dr, 5),
                     share_random_positive=(dr > 0).mean())
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv('results/robust_composite_weights.csv', index=False)
    return t


def corrected_ttest():
    runs = pd.read_csv('results/revision_runs.csv')
    rows = []
    for fs in ['orig6', 'rich']:
        for a, b in ar.CONTRASTS:
            ra = runs[(runs.fs == fs) & (runs.cond == a)].set_index('seed')
            rb = runs[(runs.fs == fs) & (runs.cond == b)].set_index('seed')
            common = ra.index.intersection(rb.index)
            for metric in ['pooled_macro_auc', 'pooled_mg']:
                d = (ra.loc[common, metric] - rb.loc[common, metric]).values
                J = len(d)
                se = np.sqrt((1 / J + TEST_TRAIN) * d.var(ddof=1))
                tval = d.mean() / se
                q = tdist.ppf(0.975, J - 1)
                rows.append({'fs': fs, 'a': a, 'b': b, 'metric': metric, 'n_seeds': J,
                             'diff': d.mean(), 'ci_lo': d.mean() - q * se, 'ci_hi': d.mean() + q * se,
                             'n_seeds_a_lower': int((d < 0).sum()),
                             'p': 2 * tdist.sf(abs(tval), J - 1)})
    t = pd.DataFrame(rows)
    t['p_holm'] = t.groupby('metric')['p'].transform(lambda p: multipletests(p, method='holm')[1])
    t.to_csv('results/robust_corrected_ttest.csv', index=False)
    return t


def main():
    pd.set_option('display.width', 200)
    ct = corrected_ttest()
    print('[corrected resampled t-test]\n', ct.round(4).to_string())
    cw = composite_weights()
    print('\n[composite weights]\n', cw.groupby(['fs', 'dimension'])[
        ['delta_equal', 'delta_prevalence', 'delta_random_median', 'delta_random_p05',
         'share_random_positive']].mean().round(3).to_string())
    eq = equal_auc_null()
    print('\n[equal-AUC null]\n', eq.round(4).to_string())


if __name__ == '__main__':
    main()
