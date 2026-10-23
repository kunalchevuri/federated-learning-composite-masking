"""
Per-client AUC table for the baseline stage, from results/revision/baseline__*.npz
(revision_experiments.py baseline: raw inputs, Adam at lr 1e-3, final model, seeds 42/123/456).

A client-dimension AUC is computed only when the client's test fold has >= 5 positives and
>= 5 negatives; otherwise it is written as 0.5 and flagged not evaluable (null_calibration.py
masks these cells).

Output: results/baseline_results.csv (method, seed, client_id, round, n_test, auc_*),
with the method names used by null_calibration.py.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REV_DIR = "results/revision/"
OUT = "results/baseline_results.csv"
MIN_CLASS = 5
DIM_COLS = ['auc_sleep', 'auc_activity', 'auc_sedentary', 'auc_mental',
            'auc_metabolic', 'auc_wellbeing', 'auc_diet']
NAME = {'centralized': 'centralized', 'central_separate': 'centralized_weighted',
        'fed_separate': 'federated_separate', 'fedavg': 'fedavg', 'scaffold_adam': 'scaffold',
        'csagg': 'csagg', 'ditto': 'ditto', 'qfedavg': 'qfedavg', 'seqlocal': 'seqlocal_fedavg'}


def main():
    rows = []
    for f in sorted(os.listdir(REV_DIR)):
        m = re.match(r'baseline__orig6_raw__(\w+?)__s(\d+)__lr', f)
        if not m or not f.endswith('.npz'):
            continue
        cond, seed = m.group(1), int(m.group(2))
        z = np.load(REV_DIR + f)
        y, p, cid = z['y_test'], z['p_test_final'], z['client_test']
        for c in range(20):
            k = cid == c
            row = {'method': NAME[cond], 'seed': seed, 'client_id': c, 'round': 0,
                   'n_test': int(k.sum())}
            for d, col in enumerate(DIM_COLS):
                yc = y[k, d]
                pos = int(yc.sum())
                ok = pos >= MIN_CLASS and len(yc) - pos >= MIN_CLASS
                row[col] = roc_auc_score(yc, p[k, d]) if ok else 0.5
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(['method', 'seed', 'client_id'])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {out.shape}; runs per method:",
          out.groupby('method')['seed'].nunique().to_dict())


if __name__ == '__main__':
    main()
