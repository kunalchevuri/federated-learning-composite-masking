"""
Analysis of the full-protocol experiments (results/revision/*.npz from
revision_experiments.py).

For every run (feature set x condition x seed) it computes, on the run's test fold:

  client level   per-client AUC_d on evaluable cells (>= 5 positives and
                 >= 5 negatives), MG, and an exact label-permutation null:
                 test rows are permuted jointly across the seven labels, which
                 keeps label correlation and the score distribution and breaks
                 only the score-label association
  population     pooled-test AUC_d with the same permutation null for pooled MG
  blind spot     the composite score a user sees, s_i = mean_d p_id, and its
                 ability to flag dimension d: Delta_d = AUC(p_d, y_d) - AUC(s, y_d).
                 Delta_d is signed; > 0 means the composite hides a signal the
                 dimension-specific output carries
  thresholds     per-dimension threshold tuned on validation (Youden J) vs 0.5:
                 TPR, FPR, balanced accuracy, F1
  other metrics  AUPRC (vs prevalence), Brier score, ECE (10 bins)
  fairness       AUC per income group and per age group, worst-group AUC,
                 largest gap, TPR gap at the tuned threshold

Outputs: results/revision_*.csv and figures/fig13_*.png, fig14_*.png
"""
import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi']        = 150

REV_DIR     = "results/revision/"
results_dir = "results/"
figures_dir = "figures/"
NHANES_PATH = "results/nhanes_clean.csv"

DIMS       = ['Sleep', 'Activity', 'Sedentary', 'Mental', 'Metabolic', 'Wellbeing', 'Diet']
LABEL_COLS = ['sleep_poor', 'activity_poor', 'sedentary_poor', 'mental_poor',
              'metabolic_poor', 'wellbeing_poor', 'diet_poor']
MIN_CLASS  = 5
N_PERM     = 1000
N_BOOT     = 2000

FS_LABEL = {'orig6_raw': 'Six inputs, raw (baseline)',
            'orig6':     'Six inputs, preprocessed',
            'rich':      'Enriched features'}
COND_LABEL = {
    'centralized':       'C1 Centralized',
    'centralized_pw':    'C1 + class weights',
    'centralized_focal': 'C1 + focal loss',
    'fedavg':            'C3 FedAvg',
    'fedavg_pw':         'C3 + class weights',
    'fedavg_focal':      'C3 + focal loss',
    'fed_separate':      'C2-FL Fed. separate',
    'central_separate':  'C2 Sep. classifiers',
    'scaffold':          'C4 SCAFFOLD',
    'csagg':             'C5 CS-Agg',
    'ditto':             'C6 Ditto',
    'qfedavg':           'C7 q-FedAvg',
    'seqlocal':          'C8 Per-dim. steps',
    'fedprox':           'C9 FedProx',
}
COND_ORDER = list(COND_LABEL)
FS_ORDER   = ['orig6_raw', 'orig6', 'rich']

# prespecified paired contrasts within a feature set (a - b), seeds as units
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

INK      = '#1F2933'
NULL_COL = '#8C959F'
OBS_COL  = '#2A6FB0'
ALT_COL  = '#C2571A'

rng = np.random.default_rng(7)


# ── helpers ────────────────────────────────────────────────────────────────────
def evaluable(y):
    k = y.sum(0)
    return (k >= MIN_CLASS) & (len(y) - k >= MIN_CLASS)


def auc_from_ranks(y, R):
    """AUC per column from average ranks R; y binary (n, d)."""
    n, k = len(y), y.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((y * R).sum(0) - k * (k + 1) / 2) / (k * (n - k))


def perm_null_mg(y, p, ok, n_perm=N_PERM):
    """Observed MG and its joint label-permutation null over evaluable dims `ok`."""
    y, p = y[:, ok].astype(np.float32), p[:, ok]
    R = rankdata(p, axis=0).astype(np.float32)
    obs = auc_from_ranks(y, R)
    perms = np.argsort(rng.random((n_perm, len(y))), axis=1)
    k, n = y.sum(0), len(y)
    S = np.einsum('bnd,nd->bd', y[perms], R)
    null = (S - k * (k + 1) / 2) / (k * (n - k))
    mg = lambda a: a.mean(-1) - a.min(-1)
    return obs, mg(obs), mg(null)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def rates(y, yhat):
    tp = ((yhat == 1) & (y == 1)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    fp = ((yhat == 1) & (y == 0)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    return tpr, fpr, (tpr + 1 - fpr) / 2, f1


def seed_ci(v):
    v = np.asarray(v, float)
    b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


# ── per-run analysis ───────────────────────────────────────────────────────────
def analyze_run(path, nh_groups):
    m = re.match(r'(eval|baseline|part-[a-z_]+)__(\w+?)__([\w.]+)__s(\d+)__lr', os.path.basename(path))
    _, fs, cond, seed = m.groups()
    # one tuned variant per feature set: FedProx mu, Ditto lambda, q-FedAvg q, CS-Agg warm-up
    cond = re.sub(r'^(fedprox)_mu[\d.]+$|^(ditto)_lam[\d.]+$|^(qfedavg)_q[\d.]+$|^(csagg)_w\d+$',
                  lambda mm: next(g for g in mm.groups() if g), cond)
    z = np.load(path)
    y, p, cid = z['y_test'], z['p_test_best'], z['client_test']
    seqn = z['seqn_test']
    run = {'fs': fs, 'cond': cond, 'seed': int(seed)}
    out = {'run': run, 'client': [], 'dim': [], 'group': []}

    # client level
    for c in range(int(cid.max()) + 1):
        msk = cid == c
        ok = evaluable(y[msk])
        auc, mg, null = perm_null_mg(y[msk], p[msk], ok)
        out['client'].append({**run, 'client_id': c, 'n_test': int(msk.sum()), 'n_dims': int(ok.sum()),
                              'mg': mg, 'mg_null_mean': null.mean(),
                              'mg_null_p95': np.percentile(null, 95),
                              'null_pctile': (null < mg).mean(),
                              **{f'auc_{DIMS[d].lower()}': a for d, a in zip(np.where(ok)[0], auc)}})

    # population (pooled test fold)
    ok = evaluable(y)
    auc, mg, null = perm_null_mg(y, p, ok)
    run.update(pooled_mg=mg, pooled_mg_null_mean=null.mean(), pooled_mg_null_p95=np.percentile(null, 95),
               pooled_mg_p=((null >= mg).sum() + 1) / (N_PERM + 1),
               pooled_macro_auc=auc.mean(), pooled_min_auc=auc.min(),
               pooled_worst_dim=DIMS[int(np.argmin(auc))])

    # composite blind spot, thresholds, other metrics
    s = p.mean(1)
    have_val = z['y_val'].size > 0
    for d in range(7):
        yd, pd_ = y[:, d], p[:, d]
        auc_d, auc_s = roc_auc_score(yd, pd_), roc_auc_score(yd, s)
        row = {**{k: run[k] for k in ('fs', 'cond', 'seed')}, 'dimension': DIMS[d],
               'prevalence': yd.mean(), 'auc': auc_d, 'auc_composite': auc_s,
               'blind_spot': auc_d - auc_s, 'auprc': average_precision_score(yd, pd_),
               'brier': np.mean((pd_ - yd) ** 2), 'ece': ece(yd, pd_)}
        row['tpr_05'], row['fpr_05'], row['bacc_05'], row['f1_05'] = rates(yd, (pd_ >= 0.5).astype(int))
        if have_val:
            thr = youden(z['y_val'][:, d], z['p_val_best'][:, d])
            yhat = (pd_ >= thr).astype(int)
            row['thr_tuned'] = thr
            row['tpr_tuned'], row['fpr_tuned'], row['bacc_tuned'], row['f1_tuned'] = rates(yd, yhat)
        else:
            yhat = (pd_ >= 0.5).astype(int)
        # subgroup AUC and TPR by income and age group
        for gname in ('income_group', 'age_group'):
            g = nh_groups.loc[seqn, gname].values
            for gv in np.unique(g):
                gm = g == gv
                if evaluable(yd[gm][:, None])[0]:
                    out['group'].append({**{k: run[k] for k in ('fs', 'cond', 'seed')},
                                         'dimension': DIMS[d], 'grouping': gname, 'group': int(gv),
                                         'n': int(gm.sum()), 'auc': roc_auc_score(yd[gm], pd_[gm]),
                                         'tpr': rates(yd[gm], yhat[gm])[0]})
        out['dim'].append(row)
    return out


# ══════════════════════════════════════════════════════════════════════════════
def main():
    nh = pd.read_csv(NHANES_PATH)
    nh['SEQN'] = nh['SEQN'].astype(int)
    nh['income_group'] = pd.cut(nh['INDFMPIR'], [0, 1.0, 2.0, 3.0, 4.0, 99], labels=False, right=False)
    nh['age_group'] = pd.cut(nh['RIDAGEYR'], [18, 36, 51, 65, 120], labels=False, right=False)
    nh_groups = nh.set_index('SEQN')[['income_group', 'age_group']]

    # baseline stage: C1 and C3 enter the pooled tables (raw inputs, final model)
    paths = sorted(glob.glob(REV_DIR + 'eval__*.npz') + glob.glob(REV_DIR + 'baseline__*__centralized__*.npz')
                   + glob.glob(REV_DIR + 'baseline__*__fedavg__*.npz'))
    print(f"{len(paths)} runs")
    runs, clients, dims, groups = [], [], [], []
    for path in paths:
        o = analyze_run(path, nh_groups)
        runs.append(o['run']); clients += o['client']; dims += o['dim']; groups += o['group']
    runs, clients = pd.DataFrame(runs), pd.DataFrame(clients)
    dims, groups = pd.DataFrame(dims), pd.DataFrame(groups)
    clients['mg_excess'] = clients['mg'] - clients['mg_null_mean']
    for name, t in [('runs', runs), ('clients', clients), ('dims', dims), ('groups', groups)]:
        t.to_csv(f"{results_dir}revision_{name}.csv", index=False)

    key = ['fs', 'cond']
    order = lambda t: t.assign(_f=t['fs'].map(FS_ORDER.index), _c=t['cond'].map(COND_ORDER.index)) \
                       .sort_values(['_f', '_c']).drop(columns=['_f', '_c'])

    # ── table A: client-level MG vs permutation null ─────────────────────────
    rows = []
    for (fs, cond), g in clients.groupby(key):
        per_client = g.groupby('client_id')['mg_excess'].mean()
        lo, hi = seed_ci(per_client)          # clients as units
        _, pw = wilcoxon(per_client)
        rows.append({'fs': fs, 'cond': cond, 'n_seeds': g['seed'].nunique(),
                     'mg': g['mg'].mean(), 'mg_null': g['mg_null_mean'].mean(),
                     'mg_excess': per_client.mean(), 'ci_lo': lo, 'ci_hi': hi, 'wilcoxon_p': pw,
                     'frac_ge_null_p95': (g['mg'] >= g['mg_null_p95']).mean(),
                     'mean_null_pctile': g['null_pctile'].mean()})
    ta = order(pd.DataFrame(rows))
    ta['p_holm'] = multipletests(ta['wilcoxon_p'], method='holm')[1]
    ta.to_csv(f"{results_dir}revision_table_client_mg.csv", index=False)
    print("\n[A] Client-level MG vs label-permutation null (clients as units)")
    print(ta.round(3).to_string(index=False))

    # ── table B: population level ────────────────────────────────────────────
    rows = []
    for (fs, cond), g in runs.groupby(key):
        r = {'fs': fs, 'cond': cond, 'n_seeds': len(g)}
        for col in ['pooled_macro_auc', 'pooled_min_auc', 'pooled_mg', 'pooled_mg_null_mean',
                    'pooled_mg_null_p95']:
            r[col] = g[col].mean()
        r['frac_seeds_mg_p<0.05'] = (g['pooled_mg_p'] < 0.05).mean()
        r['worst_dim_mode'] = g['pooled_worst_dim'].mode().iloc[0]
        rows.append(r)
    tb = order(pd.DataFrame(rows))
    tb.to_csv(f"{results_dir}revision_table_population.csv", index=False)
    print("\n[B] Population level (pooled test fold, mean over seeds)")
    print(tb.round(3).to_string(index=False))

    # ── table C: per-dimension metrics (mean over seeds) ─────────────────────
    metric_cols = ['auc', 'auc_composite', 'blind_spot', 'auprc', 'prevalence', 'brier', 'ece',
                   'tpr_05', 'tpr_tuned', 'fpr_tuned', 'bacc_05', 'bacc_tuned', 'f1_tuned']
    metric_cols = [c for c in metric_cols if c in dims]   # baseline runs have no validation
    for c in ['tpr_tuned', 'fpr_tuned', 'bacc_tuned', 'f1_tuned']:
        if c not in dims:
            dims[c] = np.nan
    metric_cols = list(dict.fromkeys(metric_cols + ['tpr_tuned', 'fpr_tuned', 'bacc_tuned', 'f1_tuned']))
    tc = dims.groupby(key + ['dimension'])[metric_cols].mean().reset_index()
    ci = dims.groupby(key + ['dimension']).agg(
        auc_ci=('auc', lambda v: seed_ci(v)), blind_ci=('blind_spot', lambda v: seed_ci(v))).reset_index()
    tc = tc.merge(ci, on=key + ['dimension'])
    tc['auc_lo'] = tc['auc_ci'].str[0]; tc['auc_hi'] = tc['auc_ci'].str[1]
    tc['blind_lo'] = tc['blind_ci'].str[0]; tc['blind_hi'] = tc['blind_ci'].str[1]
    tc = order(tc.drop(columns=['auc_ci', 'blind_ci']))
    tc.to_csv(f"{results_dir}revision_table_dims.csv", index=False)
    print("\n[C] Per-dimension pooled AUC (mean over seeds; * = seed CI above 0.5)")
    tc['cell'] = tc['auc'].map('{:.3f}'.format) + np.where(tc['auc_lo'] > 0.5, '*', ' ')
    print(tc.pivot_table(index=key, columns='dimension', values='cell', aggfunc='first')[DIMS]
          .pipe(lambda t: t.loc[[i for i in [(f, c) for f in FS_ORDER for c in COND_ORDER] if i in t.index]])
          .to_string())
    print("\n[C2] Composite blind spot Delta_d = AUC(p_d) - AUC(composite) (* = CI excludes 0)")
    tc['bcell'] = tc['blind_spot'].map('{:+.3f}'.format) + np.where(
        (tc['blind_lo'] > 0) | (tc['blind_hi'] < 0), '*', ' ')
    print(tc.pivot_table(index=key, columns='dimension', values='bcell', aggfunc='first')[DIMS]
          .pipe(lambda t: t.loc[[i for i in [(f, c) for f in FS_ORDER for c in COND_ORDER] if i in t.index]])
          .to_string())
    print("\n[C3] Sensitivity (TPR) at 0.5 vs validation-tuned threshold")
    tc['tcell'] = tc['tpr_05'].map('{:.2f}'.format) + '/' + tc['tpr_tuned'].map('{:.2f}'.format)
    print(tc.pivot_table(index=key, columns='dimension', values='tcell', aggfunc='first')[DIMS]
          .pipe(lambda t: t.loc[[i for i in [(f, c) for f in FS_ORDER for c in COND_ORDER] if i in t.index]])
          .to_string())

    # ── table D: paired contrasts across seeds ───────────────────────────────
    rows = []
    for fs in ['orig6', 'rich']:
        for a, b in CONTRASTS:
            ra = runs[(runs.fs == fs) & (runs.cond == a)].set_index('seed')
            rb = runs[(runs.fs == fs) & (runs.cond == b)].set_index('seed')
            common = ra.index.intersection(rb.index)
            if len(common) < 5:
                continue
            for metric in ['pooled_macro_auc', 'pooled_min_auc', 'pooled_mg']:
                d = (ra.loc[common, metric] - rb.loc[common, metric]).values
                lo, hi = seed_ci(d)
                rows.append({'fs': fs, 'contrast': f"{COND_LABEL[a]} - {COND_LABEL[b]}",
                             'metric': metric, 'mean_diff': d.mean(), 'ci_lo': lo, 'ci_hi': hi,
                             'n_seeds': len(d), 'wilcoxon_p': wilcoxon(d).pvalue})
            da = dims[(dims.fs == fs) & (dims.cond == a)].set_index(['seed', 'dimension'])['auc']
            db = dims[(dims.fs == fs) & (dims.cond == b)].set_index(['seed', 'dimension'])['auc']
            for dname in DIMS:
                d = (da.xs(dname, level='dimension') - db.xs(dname, level='dimension')).dropna().values
                lo, hi = seed_ci(d)
                rows.append({'fs': fs, 'contrast': f"{COND_LABEL[a]} - {COND_LABEL[b]}",
                             'metric': f'auc_{dname}', 'mean_diff': d.mean(), 'ci_lo': lo, 'ci_hi': hi,
                             'n_seeds': len(d), 'wilcoxon_p': wilcoxon(d).pvalue})
    td = pd.DataFrame(rows)
    if len(td):
        td['p_holm'] = multipletests(td['wilcoxon_p'], method='holm')[1]
        td.to_csv(f"{results_dir}revision_table_contrasts.csv", index=False)
        print("\n[D] Paired contrasts over seeds (Holm across all rows)")
        print(td[td['metric'].str.startswith('pooled')].round(4).to_string(index=False))

    # ── table E: subgroup fairness ───────────────────────────────────────────
    fg = groups.groupby(key + ['dimension', 'grouping', 'seed']).agg(
        worst=('auc', 'min'), best=('auc', 'max'), tpr_gap=('tpr', lambda v: v.max() - v.min()))
    fg['gap'] = fg['best'] - fg['worst']
    te = fg.groupby(key + ['dimension', 'grouping'])[['worst', 'gap', 'tpr_gap']].mean().reset_index()
    te = order(te)
    te.to_csv(f"{results_dir}revision_table_fairness.csv", index=False)
    print("\n[E] Subgroup AUC gap (income groups), mean over seeds")
    print(te[te.grouping == 'income_group'].pivot_table(index=key, columns='dimension', values='gap')[DIMS]
          .round(3).to_string())

    # ── label-level composite masking (data only) ────────────────────────────
    L = nh[LABEL_COLS].values
    comp = 1 - L.mean(1)           # share of dimensions not flagged poor
    med = np.median(comp)
    rows = [{'dimension': d, 'n_poor': int(L[:, i].sum()),
             'share_composite_at_or_above_median': float((comp[L[:, i] == 1] >= med).mean())}
            for i, d in enumerate(DIMS)]
    tl = pd.DataFrame(rows)
    tl.to_csv(f"{results_dir}revision_label_masking.csv", index=False)
    print(f"\n[F] Label-level masking: adults poor on a dimension whose unweighted composite "
          f"(share of dimensions not poor) is at or above the population median ({med:.3f})")
    print(tl.round(3).to_string(index=False))

    make_figures(tc)


def make_figures(tc):
    # Fig 13: per-dimension pooled AUC, C1 vs C3, three feature regimes
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True)
    ypos = np.arange(len(DIMS))[::-1]
    for ax, fs in zip(axes, FS_ORDER):
        ax.axvline(0.5, color=NULL_COL, lw=1, ls='--')
        for cond, col, off, mk in [('centralized', INK, 0.15, 'o'), ('fedavg', OBS_COL, -0.15, 's')]:
            t = tc[(tc.fs == fs) & (tc.cond == cond)].set_index('dimension').reindex(DIMS)
            if t['auc'].isna().all():
                continue
            ax.hlines(ypos + off, t['auc_lo'], t['auc_hi'], color=col, lw=1.8)
            ax.plot(t['auc'], ypos + off, mk, color=col, ms=4.5, mec='white', mew=0.8,
                    label=COND_LABEL[cond])
        ax.set_title(FS_LABEL[fs].replace(' (baseline)', '\n(baseline)'),
                     fontsize=8, loc='left', color=INK)
        ax.set_xlim(0.4, 0.85)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel('Pooled test AUC', fontsize=8)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(DIMS, fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.07))
    plt.tight_layout()
    path = f"{figures_dir}fig13_dimension_auc_by_regime.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> {path}")

    # Fig 14: composite blind spot, enriched features, C1 vs C3
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axvline(0, color=NULL_COL, lw=1, ls='--')
    for cond, col, off, mk in [('centralized', INK, 0.15, 'o'), ('fedavg', OBS_COL, -0.15, 's')]:
        t = tc[(tc.fs == 'rich') & (tc.cond == cond)].set_index('dimension').reindex(DIMS)
        ax.hlines(ypos + off, t['blind_lo'], t['blind_hi'], color=col, lw=1.8)
        ax.plot(t['blind_spot'], ypos + off, mk, color=col, ms=4.5, mec='white', mew=0.8,
                label=COND_LABEL[cond])
    ax.set_yticks(ypos)
    ax.set_yticklabels(DIMS, fontsize=8)
    ax.set_xlabel('AUC(dimension output) − AUC(composite score)', fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    plt.tight_layout()
    path = f"{figures_dir}fig14_composite_blind_spot.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  -> {path}")


if __name__ == '__main__':
    main()
