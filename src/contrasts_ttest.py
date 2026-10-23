"""
Paired contrasts between conditions of the revised protocol, seeds as units:
paired t-test, t-based 95% CI, Holm correction within each metric (across
feature sets and contrasts).

Input: results/revision_runs.csv (analyze_revision.py)
Output: results/revision_contrasts_ttest.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

CONTRASTS = [
    ('fed_separate',      'fedavg'),
    ('fedavg_pw',         'fedavg'),
    ('fedavg_focal',      'fedavg'),
    ('centralized_pw',    'centralized'),
    ('centralized_focal', 'centralized'),
    ('fedavg',            'centralized'),
    ('central_separate',  'centralized'),
    ('scaffold',          'fedavg'),
    ('csagg',             'fedavg'),
    ('ditto',             'fedavg'),
    ('qfedavg',           'fedavg'),
    ('seqlocal',          'fedavg'),
    ('fedprox',           'fedavg'),
]
METRICS = ['pooled_macro_auc', 'pooled_min_auc', 'pooled_mg']


def contrasts(runs, pairs):
    rows = []
    for fs in ['orig6', 'rich']:
        for a, b in pairs:
            ra = runs[(runs.fs == fs) & (runs.cond == a)].set_index('seed')
            rb = runs[(runs.fs == fs) & (runs.cond == b)].set_index('seed')
            common = ra.index.intersection(rb.index)
            if len(common) < 5:
                continue
            for m in METRICS:
                d = (ra.loc[common, m] - rb.loc[common, m]).values
                half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                rows.append({'fs': fs, 'contrast': f'{a}-{b}', 'metric': m, 'diff': d.mean(),
                             'lo': d.mean() - half, 'hi': d.mean() + half,
                             'same_sign': int((np.sign(d) == np.sign(d.mean())).sum()),  # seeds agreeing with the mean
                             'p': stats.ttest_rel(ra.loc[common, m], rb.loc[common, m]).pvalue})
    t = pd.DataFrame(rows)
    t['p_holm'] = t.groupby('metric')['p'].transform(lambda p: multipletests(p, method='holm')[1])
    return t


if __name__ == '__main__':
    runs = pd.read_csv('results/revision_runs.csv')
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        # the first six contrasts must reproduce the file used for the current paper
        old = pd.read_csv('results/revision_contrasts_ttest.csv')
        new = contrasts(runs, CONTRASTS[:6])
        cols = ['diff', 'lo', 'hi', 'same_sign', 'p', 'p_holm']
        print('max abs difference:', np.abs(new[cols].values - old[cols].values).max())
    else:
        t = contrasts(runs, CONTRASTS)
        t.to_csv('results/revision_contrasts_ttest.csv', index=False)
        pd.set_option('display.width', 200)
        print(t.round(4).to_string(index=False))
