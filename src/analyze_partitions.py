"""
Partition robustness: are the client-level null result and the composite blind
spot artifacts of the income x age federation?

For each client partition (income x age as in the paper; a fixed random IID
split into 20 clients; race/ethnicity x sex, 12 clients) and feature set:
  * FedAvg trained on that partition (revision_experiments.py partition;
    income x age reuses the eval runs)
  * centralized training does not depend on the partition, so its eval
    predictions are regrouped by the partition's clients
Reports per-client MG against the label-permutation null (clients as units,
cluster bootstrap CI), pooled macro-AUC and the composite blind spot.

Output: results/revision_partitions.csv
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_revision as ar  # noqa: E402
import revision_experiments as rx  # noqa: E402

REV = "results/revision/"
rng = np.random.default_rng(3)


def summarize(y, p, cid):
    clients = []
    for c in np.unique(cid):
        m = cid == c
        ok = ar.evaluable(y[m])
        _, mg, null = ar.perm_null_mg(y[m], p[m], ok)
        clients.append({'client': int(c), 'n_test': int(m.sum()), 'mg': mg, 'null': null.mean(),
                        'ge_p95': mg >= np.percentile(null, 95)})
    s = p.mean(1)
    blind = [roc_auc_score(y[:, d], p[:, d]) - roc_auc_score(y[:, d], s) for d in range(7)]
    auc = [roc_auc_score(y[:, d], p[:, d]) for d in range(7)]
    return pd.DataFrame(clients), np.mean(auc), np.array(blind)


def main():
    df = rx.load_data()
    seqn = df['SEQN'].values
    runs = []
    for part in ['income_age'] + rx.PARTITIONS:
        cid_of = dict(zip(seqn, rx.partition_ids(df, part)))
        for fs in rx.FEATURE_SETS:
            for cond in ['centralized', 'fedavg']:
                if cond == 'fedavg' and part != 'income_age':
                    paths = glob.glob(f"{REV}part-{part}__{fs}__fedavg__s*__lr*.npz")
                else:
                    paths = glob.glob(f"{REV}eval__{fs}__{cond}__s*__lr*.npz")
                for path in sorted(paths):
                    seed = int(re.search(r'__s(\d+)__', path).group(1))
                    z = np.load(path)
                    cid = np.array([cid_of[s] for s in z['seqn_test']])
                    if 'part-' in path or part == 'income_age':
                        assert np.array_equal(cid, z['client_test']), path
                    cl, macro, blind = summarize(z['y_test'], z['p_test_best'], cid)
                    runs.append({'partition': part, 'fs': fs, 'cond': cond, 'seed': seed,
                                 'clients': cl, 'macro_auc': macro, 'blind': blind})
    rows = []
    for (part, fs, cond), g in pd.DataFrame(runs).groupby(['partition', 'fs', 'cond'], sort=False):
        cl = pd.concat([r.assign(seed=s) for r, s in zip(g['clients'], g['seed'])])
        exc = (cl['mg'] - cl['null']).groupby(cl['client']).mean()        # clients as units
        boot = exc.values[rng.integers(0, len(exc), (5000, len(exc)))].mean(1)
        blind = np.vstack(g['blind'].values).mean(0)
        rows.append({'partition': part, 'fs': fs, 'cond': cond, 'n_seeds': len(g),
                     'n_clients': len(exc), 'test_min': int(cl['n_test'].min()),
                     'test_max': int(cl['n_test'].max()),
                     'mg': cl['mg'].mean(), 'mg_null': cl['null'].mean(), 'mg_excess': exc.mean(),
                     'ci_lo': np.percentile(boot, 2.5), 'ci_hi': np.percentile(boot, 97.5),
                     'frac_ge_p95': cl['ge_p95'].mean(), 'macro_auc': g['macro_auc'].mean(),
                     'blind_min': blind.min(), 'blind_max': blind.max()})
    out = pd.DataFrame(rows)
    out.to_csv("results/revision_partitions.csv", index=False)
    pd.set_option('display.width', 220)
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
