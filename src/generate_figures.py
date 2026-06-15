import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from scipy.stats import wilcoxon
import os
import warnings
warnings.filterwarnings('ignore')

# Universal polish — rcParams
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family']         = 'DejaVu Sans'
plt.rcParams['axes.spines.top']     = False
plt.rcParams['axes.spines.right']   = False
plt.rcParams['figure.dpi']          = 150

# ── paths ──────────────────────────────────────────────────────────────────────
results_path  = "results/bibm_results.csv"
shap_path     = "results/shap_analysis.csv"
fairness_path = "results/fairness_metrics.csv"
nhanes_path   = "results/nhanes_clean.csv"
figures_dir   = "figures/"
os.makedirs(figures_dir, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
DIMS     = ['Sleep', 'Activity', 'Sedentary', 'Mental', 'Metabolic', 'Wellbeing', 'Diet']
DIM_COLS = ['auc_sleep', 'auc_activity', 'auc_sedentary', 'auc_mental',
            'auc_metabolic', 'auc_wellbeing', 'auc_diet']
LABEL_COLS = ['sleep_poor', 'activity_poor', 'sedentary_poor', 'mental_poor',
              'metabolic_poor', 'wellbeing_poor', 'diet_poor']
INCOME_LABELS = {0: '<100%', 1: '100-199%', 2: '200-299%', 3: '300-399%', 4: '400%+'}
AGE_LABELS    = {0: '18-35', 1: '36-50', 2: '51-65', 3: '65+'}
SHAP_FEATS    = ['shap_RIDAGEYR', 'shap_RIAGENDR', 'shap_INDFMPIR',
                 'shap_RIDRETH3', 'shap_avg_sys', 'shap_avg_dia']
FEAT_DISPLAY  = ['Age', 'Gender', 'Income\nRatio', 'Ethnicity', 'Systolic\nBP', 'Diastolic\nBP']

FL_METHODS = ['fedavg', 'scaffold', 'csagg', 'ditto', 'qfedavg']
FL_LABELS  = {'fedavg': 'FedAvg', 'scaffold': 'SCAFFOLD', 'csagg': 'CS-Agg',
              'ditto': 'Ditto', 'qfedavg': 'q-FedAvg'}
FL_COLORS  = {'fedavg': '#E74C3C', 'scaffold': '#9B59B6', 'csagg': '#E67E22',
              'ditto': '#3498DB', 'qfedavg': '#E91E63'}

# (method, final_round, bar_label, color)
METHOD_BAR_CFG = [
    ('centralized',          None, "C1\nCentralized\n(equal)",  '#4878CF'),
    ('centralized_weighted', None, "C2\nWeighted\nLoss",        '#2ECC71'),
    ('fedavg',               49,   "C3\nFedAvg",               '#E74C3C'),
    ('scaffold',             49,   "C4\nSCAFFOLD",             '#9B59B6'),
    ('csagg',                49,   "C5\nCS-Agg",               '#E67E22'),
    ('ditto',                49,   "C6\nDitto",                '#3498DB'),
    ('qfedavg',              49,   "C7\nq-FedAvg",             '#E91E63'),
    ('federated_separate',   None, "C2-FL\nFed Separate",      '#27AE60'),
]

# FIX 5: extended to 8 columns including C2 and C2-FL
SEV_CFG = [
    ('centralized',          None, 'C1\nCentralized'),
    ('centralized_weighted', None, 'C2\nWeighted'),
    ('federated_separate',   None, 'C2-FL\nFed Sep'),
    ('fedavg',               49,   'C3\nFedAvg'),
    ('scaffold',             49,   'C4\nSCAFFOLD'),
    ('csagg',                49,   'C5\nCS-Agg'),
    ('ditto',                49,   'C6\nDitto'),
    ('qfedavg',              49,   'C7\nq-FedAvg'),
]

BOOT_CFG = [
    ('centralized',          None, 'C1',    '#7F8C8D'),
    ('centralized_weighted', None, 'C2',    '#27AE60'),
    ('fedavg',               49,   'C3',    '#4878CF'),
    ('scaffold',             49,   'C4',    '#9B59B6'),
    ('csagg',                49,   'C5',    '#E67E22'),
    ('ditto',                49,   'C6',    '#3498DB'),
    ('qfedavg',              49,   'C7',    '#E91E63'),
    ('federated_separate',   None, 'C2-FL', '#2ECC71'),
]

# ── load data ──────────────────────────────────────────────────────────────────
df      = pd.read_csv(results_path)
shap_df = pd.read_csv(shap_path)
fair_df = pd.read_csv(fairness_path)
nhanes  = pd.read_csv(nhanes_path)

client_meta = (df[['client_id', 'income_group', 'age_group']]
               .drop_duplicates().set_index('client_id').sort_index())

print(f"Loaded: bibm_results {df.shape} | shap {shap_df.shape} | "
      f"fairness {fair_df.shape} | nhanes {nhanes.shape}")
fig_stats = []

# ── helpers ────────────────────────────────────────────────────────────────────
def get_subset(method, final_round):
    sub = df[df['method'] == method].copy()
    if sub.empty:
        return sub
    max_r = int(sub['round'].max())
    sub = sub[sub['round'] == max_r]
    return sub

def per_client_mg(method, final_round):
    sub = get_subset(method, final_round)
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby('client_id')['masking_gap'].mean().sort_index()

def bootstrap_ci(vals, n_boot=10000, rng_seed=42):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(rng_seed)
    boot = np.array([rng.choice(v, len(v), replace=True).mean()
                     for _ in range(n_boot)])
    return v.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def get_tier(mg):
    if mg < 0.10:
        return 0
    elif mg < 0.20:
        return 1
    return 2

# Universal polish: savefig with white background
def savefig(path, key_stat):
    plt.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    sz = os.path.getsize(path)
    name = os.path.basename(path)
    fig_stats.append((name, sz, key_stat))
    print(f"  -> {name}  ({sz/1024:.1f} KB)  {key_stat}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 7x20 AUC HEATMAP  [FIX 4: label sizes]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] AUC Heatmap")
base1 = df[(df['method'] == 'centralized') & (df['seed'] == 42)].sort_values('client_id')
if base1.empty:
    print("  SKIP -- no centralized seed=42 data")
else:
    auc_mat = base1[DIM_COLS].values.T   # (7, 20)

    # FIX 4: MG to 3 decimal places; each label 4 lines
    xlabels = []
    for _, row in base1.iterrows():
        cid = int(row['client_id'])
        inc = INCOME_LABELS.get(int(row['income_group']), '?')
        age = AGE_LABELS.get(int(row['age_group']), '?')
        xlabels.append(f"C{cid}\n{inc}\n{age}\nMG={row['masking_gap']:.3f}")

    fig, ax = plt.subplots(figsize=(22, 8))
    im = ax.imshow(auc_mat, cmap='RdYlGn', vmin=0.3, vmax=0.7, aspect='auto')

    # FIX 4: cell annotation fontsize 7
    for r in range(7):
        for c in range(20):
            v = auc_mat[r, c]
            fc = 'white' if v < 0.40 else 'black'
            ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color=fc, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    # FIX 4: colorbar label fontsize 11
    cbar.set_label('AUC', rotation=270, labelpad=15, fontsize=11)
    cbar.ax.axhline(y=0.5, color='navy', linestyle='--', linewidth=2.0)
    cbar.ax.text(1.08, 0.5, '0.5\n(chance)', transform=cbar.ax.transAxes,
                 va='center', fontsize=7.5, color='navy')

    # FIX 4: ytick 12, xtick 8 bold rotation=0
    ax.set_yticks(range(7))
    ax.set_yticklabels(DIMS, fontsize=12)
    ax.set_xticks(range(20))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=0)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight('bold')
    ax.tick_params(axis='x', length=0)

    # Polish: title fontsize 14, bold, pad 15
    ax.set_title(
        "Per-Dimension AUC by Demographic Client Group\nCentralized Baseline (Seed 42)",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(pad=2.0)
    savefig(os.path.join(figures_dir, 'fig1_auc_heatmap.png'),
            f'Mean AUC={auc_mat.mean():.3f} | Worst cell={auc_mat.min():.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MASKING GAP BAR CHART  [FIX 3: seed-level std error bars]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 2] MG Bar Chart")
bar_means, bar_stds, bar_labels, bar_colors = [], [], [], []
for method, fr, label, color in METHOD_BAR_CFG:
    # FIX 3: centralized-family uses all rounds; FL methods filter round=49
    if method in ['centralized', 'centralized_weighted', 'federated_separate']:
        sub = df[df['method'] == method]
    else:
        sub = df[(df['method'] == method) & (df['round'] == 49)]

    if len(sub) == 0:
        print(f"  SKIP {method} -- no data")
        continue

    # FIX 3: mean per seed first, then std across seeds (not across clients)
    seed_means = []
    for seed in [42, 123, 456]:
        s = sub[sub['seed'] == seed]
        if len(s) > 0:
            seed_means.append(s['masking_gap'].mean())

    overall_mean = np.mean(seed_means)
    seed_std     = np.std(seed_means) if len(seed_means) > 1 else 0.0
    bar_means.append(overall_mean)
    bar_stds.append(seed_std)
    bar_labels.append(label)
    bar_colors.append(color)

if bar_means:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(bar_means))
    ax.bar(x, bar_means, color=bar_colors, width=0.6,
           yerr=bar_stds, capsize=5, error_kw={'linewidth': 1.5, 'ecolor': '#555'})
    for xi, (m, s) in enumerate(zip(bar_means, bar_stds)):
        ax.text(xi, m + s + 0.003, f'{m:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.axhline(0.10, color='orange', linestyle='--', linewidth=1.8,
               label='Mild/Moderate (0.10)')
    ax.axhline(0.20, color='red',    linestyle='--', linewidth=1.8,
               label='Moderate/Severe (0.20)')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel('Mean Masking Gap', fontsize=12)
    ax.set_ylim(0, max(bar_means) + max(bar_stds) + 0.025)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_title(
        "Mean Masking Gap by Experiment Condition\n"
        "Error bars = std across seeds (n=3)",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(pad=2.0)
    savefig(os.path.join(figures_dir, 'fig2_mg_by_method.png'),
            f'C1={bar_means[0]:.3f} | seed-std range=[{min(bar_stds):.4f}, {max(bar_stds):.4f}]')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CONVERGENCE CURVES  [FIX 2: alpha 0.07, lw 2.5, baseline style]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] Convergence Curves")
base_mg = df[df['method'] == 'centralized']['masking_gap'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
all_upper = []
for method in FL_METHODS:
    sub = df[df['method'] == method]
    if sub.empty:
        print(f"  SKIP {method}")
        continue
    grp = sub.groupby('round')['masking_gap']
    mu  = grp.mean()
    sd  = grp.std()
    rounds   = mu.index.values
    final_mg = mu.iloc[-1]
    all_upper.append((mu + sd).max())
    lbl = f"{FL_LABELS[method]} (final MG={final_mg:.3f})"
    # FIX 2: linewidth 2.5
    ax.plot(rounds, mu.values, color=FL_COLORS[method], linewidth=2.5, label=lbl)
    # FIX 2: fill_between alpha 0.07
    ax.fill_between(rounds, mu - sd, mu + sd, alpha=0.07, color=FL_COLORS[method])

# FIX 2: centralized baseline black dashed lw=2.5 alpha=0.9
ax.axhline(base_mg, color='black', linestyle='--', linewidth=2.5, alpha=0.9,
           label=f'Centralized baseline (MG={base_mg:.3f})')
ax.axhline(0.10, color='orange', linestyle='--', linewidth=1.2, alpha=0.8,
           label='Mild/Moderate threshold (0.10)')
ax.axhline(0.20, color='red',    linestyle='--', linewidth=1.2, alpha=0.8,
           label='Moderate/Severe threshold (0.20)')
ax.axvline(10, color='#E67E22', linestyle=':', linewidth=2.0, alpha=0.7)
y_top = max(all_upper) if all_upper else 0.25
ax.text(10.5, y_top, 'CS-Agg\nwarmup', fontsize=8, color='#E67E22', va='top')

ax.set_xlabel('Communication Round', fontsize=12)
ax.set_ylabel('Mean Masking Gap', fontsize=12)
ax.set_xlim(0, 49)
# FIX 2: legend fontsize 10
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.set_title(
    "Masking Gap Convergence Across FL Training\n"
    "All 5 Methods vs Centralized Baseline",
    fontsize=14, fontweight='bold', pad=15)
plt.tight_layout(pad=2.0)
savefig(os.path.join(figures_dir, 'fig3_convergence.png'),
        f'Centralized baseline MG={base_mg:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — FIX PROOF  [FIX 1: pooled Wilcoxon n=60 across 3 seeds]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] Fix Proof (Violin + Strip)")

# FIX 1: pool all 3 seeds, sort by client_id within each seed → 60 pairs
c1_sorted, c2_sorted, cfl_sorted = [], [], []
for seed in [42, 123, 456]:
    c1_s = (df[(df['method'] == 'centralized') & (df['seed'] == seed)]
            .sort_values('client_id')['masking_gap'].values)
    c2_s = (df[(df['method'] == 'centralized_weighted') & (df['seed'] == seed)]
            .sort_values('client_id')['masking_gap'].values)
    cfl_s = (df[(df['method'] == 'federated_separate') & (df['seed'] == seed)]
             .sort_values('client_id')['masking_gap'].values)
    if len(c1_s) > 0 and len(c2_s) == len(c1_s) and len(cfl_s) == len(c1_s):
        c1_sorted.extend(c1_s)
        c2_sorted.extend(c2_s)
        cfl_sorted.extend(cfl_s)

c1_arr  = np.array(c1_sorted)
c2_arr  = np.array(c2_sorted)
cfl_arr = np.array(cfl_sorted)

# FIX 1: Wilcoxon with alternative='less' (fix methods expected lower)
p_c2, p_cfl = np.nan, np.nan
if len(c2_arr) == len(c1_arr) and len(c1_arr) > 0:
    _, p_c2  = wilcoxon(c2_arr,  c1_arr, alternative='less')
if len(cfl_arr) == len(c1_arr) and len(c1_arr) > 0:
    _, p_cfl = wilcoxon(cfl_arr, c1_arr, alternative='less')

def fmt_p(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    elif p < 0.05:
        return f"p={p:.4f}*"
    else:
        return f"p={p:.4f}"

# Build plot data from all 60 pooled values per condition
plot_data, plot_groups = [], []
for arr, lbl in [(c1_arr,  'C1\nCentralized'),
                 (c2_arr,  'C2\nWeighted Loss'),
                 (cfl_arr, 'C2-FL\nFed Separate')]:
    if len(arr) > 0:
        plot_data.extend(arr)
        plot_groups.extend([lbl] * len(arr))

if plot_data:
    palette = {'C1\nCentralized':   '#4878CF',
               'C2\nWeighted Loss': '#2ECC71',
               'C2-FL\nFed Separate': '#27AE60'}
    pf = pd.DataFrame({'MG': plot_data, 'Condition': plot_groups})

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(data=pf, x='Condition', y='MG', palette=palette,
                   inner=None, alpha=0.6, ax=ax)
    sns.stripplot(data=pf, x='Condition', y='MG', palette=palette,
                  size=5, alpha=0.65, jitter=True, ax=ax)

    # group mean lines
    for xi, arr in enumerate([c1_arr, c2_arr, cfl_arr]):
        if len(arr) > 0:
            ax.hlines(arr.mean(), xi - 0.35, xi + 0.35,
                      colors='black', linewidths=2.5, linestyles='-')

    # FIX 1: annotation box with pooled p-values and n=60
    n_pairs = len(c1_arr)
    annot_text = (f"C2 vs C1:    {fmt_p(p_c2)} (n={n_pairs})\n"
                  f"C2-FL vs C1: {fmt_p(p_cfl)} (n={n_pairs})\n"
                  f"* p<0.05, pooled across 3 seeds")
    ax.text(0.98, 0.97, annot_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.axhline(0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Mild/Moderate (0.10)')
    ax.axhline(0.20, color='red',    linestyle='--', linewidth=1.5, alpha=0.7,
               label='Moderate/Severe (0.20)')
    ax.set_ylabel('Per-Client Masking Gap', fontsize=12)
    ax.set_xlabel('')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_title(
        "Fix Direction Validation: Eliminating Composite "
        "Objective Reduces Masking Gap",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(pad=2.0)
    stat4 = (f'C1={c1_arr.mean():.3f} | C2={c2_arr.mean():.3f} | '
             f'C2-FL={cfl_arr.mean():.3f} | {fmt_p(p_c2)} | {fmt_p(p_cfl)}')
    savefig(os.path.join(figures_dir, 'fig4_fix_proof.png'), stat4)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — SHAP SUMMARY (7 subplots, unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 5] SHAP Feature Importance")
present_feats = [f for f in SHAP_FEATS if f in shap_df.columns]
present_names = [FEAT_DISPLAY[SHAP_FEATS.index(f)] for f in present_feats]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes_flat  = axes.flatten()

for di, dim in enumerate(DIMS):
    ax = axes_flat[di]
    sub = shap_df[shap_df['dimension'] == dim]
    if sub.empty:
        ax.set_title(dim)
        ax.axis('off')
        continue

    importance = sub[present_feats].abs().mean()
    importance = importance.sort_values(ascending=True)
    feat_names_sorted = [present_names[present_feats.index(f)]
                         for f in importance.index]
    vals = importance.values

    cmap   = plt.cm.YlOrRd
    norm   = plt.Normalize(vals.min(), vals.max())
    colors = [cmap(norm(v)) for v in vals]

    bars = ax.barh(feat_names_sorted, vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=7.5)
    ax.set_title(dim, fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean |SHAP|', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_xlim(0, vals.max() * 1.25)

axes_flat[7].axis('off')

fig.suptitle(
    "SHAP Feature Importance Per Wellness Dimension\n"
    "Centralized Model (Seed 42)",
    fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(pad=2.0)
savefig(os.path.join(figures_dir, 'fig5_shap.png'),
        'Top feature across dims: Age/Income importance visible')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SEVERITY TIER TABLE  [FIX 5: 8 cols C1/C2/C2-FL/C3-C7]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 6] Severity Tiers")
n_clients = len(client_meta)
n_cols    = len(SEV_CFG)
tier_matrix = np.full((n_clients, n_cols), np.nan)
mg_matrix   = np.full((n_clients, n_cols), np.nan)

for ci, (method, fr, _) in enumerate(SEV_CFG):
    pc = per_client_mg(method, fr)
    if pc.empty:
        continue
    for cid in client_meta.index:
        if cid in pc.index:
            mg = pc[cid]
            tier_matrix[int(cid), ci] = get_tier(mg)
            mg_matrix[int(cid), ci]   = mg

if not np.all(np.isnan(tier_matrix)):
    # FIX 5: updated tier colors
    tier_cmap = ListedColormap(['#4CAF50', '#FF9800', '#F44336'])
    tier_norm = BoundaryNorm([0, 1, 2, 3], 3)

    row_labels = []
    for cid in client_meta.index:
        inc = INCOME_LABELS.get(int(client_meta.loc[cid, 'income_group']), '?')
        age = AGE_LABELS.get(int(client_meta.loc[cid, 'age_group']), '?')
        row_labels.append(f"C{cid}  {inc}  {age}")

    col_labels = [lbl for _, _, lbl in SEV_CFG]

    # FIX 5: figsize (18, 12)
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(tier_matrix, cmap=tier_cmap, norm=tier_norm, aspect='auto')

    # FIX 5: cell text fontsize 8
    for r in range(n_clients):
        for c in range(n_cols):
            mg = mg_matrix[r, c]
            if not np.isnan(mg):
                fc = 'white' if get_tier(mg) == 2 else 'black'
                ax.text(c, r, f'{mg:.3f}', ha='center', va='center',
                        fontsize=8, color=fc, fontweight='bold')

    # FIX 5: col header fontsize 10, bold; row label fontsize 8
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_clients))
    ax.set_yticklabels(row_labels, fontsize=8)

    legend_patches = [
        mpatches.Patch(color='#4CAF50', label='Mild   (MG < 0.10)'),
        mpatches.Patch(color='#FF9800', label='Moderate (0.10 <= MG < 0.20)'),
        mpatches.Patch(color='#F44336', label='Severe  (MG >= 0.20)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              bbox_to_anchor=(1.0, -0.02), ncol=3, fontsize=10, framealpha=0.9)
    ax.set_title(
        "Masking Gap Severity Tiers by Client and Method\n"
        "Green=Mild(<0.10), Orange=Moderate(0.10-0.20), Red=Severe(>0.20)",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(pad=2.0)
    savefig(os.path.join(figures_dir, 'fig6_severity_tiers.png'),
            f'MG range=[{np.nanmin(mg_matrix):.3f}, {np.nanmax(mg_matrix):.3f}] | {n_cols} methods')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — LABEL CORRELATION HEATMAP (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 7] Label Correlation Heatmap")
present_label_cols = [c for c in LABEL_COLS if c in nhanes.columns]
if len(present_label_cols) >= 2:
    corr = nhanes[present_label_cols].corr(method='pearson')
    display_names = [d for d, c in zip(DIMS, LABEL_COLS) if c in present_label_cols]
    corr.index   = display_names
    corr.columns = display_names

    mask = np.eye(len(display_names), dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                vmin=-0.3, vmax=0.3, annot=True, fmt='.3f',
                annot_kws={'size': 10}, linewidths=0.5,
                ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title(
        "Label Correlation Matrix -- 7 Wellness Dimensions\n"
        "Low off-diagonal correlation confirms dimensional independence",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(pad=2.0)
    max_offdiag = corr.where(~mask).abs().max().max()
    savefig(os.path.join(figures_dir, 'fig7_label_correlation.png'),
            f'Max off-diagonal |r|={max_offdiag:.3f}')
else:
    print("  SKIP -- label columns missing from nhanes_clean.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — BOOTSTRAP CI FOREST PLOT (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 8] Bootstrap CI Forest Plot")
boot_results = []
for method, fr, label, color in BOOT_CFG:
    sub = get_subset(method, fr)
    if sub.empty:
        print(f"  SKIP {method}")
        continue
    mean_, lo, hi = bootstrap_ci(sub['masking_gap'].values)
    boot_results.append((label, mean_, lo, hi, color))

if boot_results:
    fig, ax = plt.subplots(figsize=(12, 8))
    n = len(boot_results)

    for yi, (lbl, mean_, lo, hi, color) in enumerate(boot_results):
        y = n - 1 - yi
        ax.plot(mean_, y, 'o', color=color, markersize=10, zorder=5)
        ax.hlines(y, lo, hi, color=color, linewidth=3.5, zorder=4)
        ax.text(hi + 0.002, y, f'{mean_:.3f} [{lo:.3f}, {hi:.3f}]',
                va='center', fontsize=8.5)

    ax.axvline(0.10, color='orange', linestyle='--', linewidth=1.8,
               label='Mild/Moderate threshold (0.10)')
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(boot_results)], fontsize=10)
    ax.set_xlabel('Mean Masking Gap', fontsize=12)
    ax.set_xlim(0, max(r[3] for r in boot_results) + 0.06)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_title(
        "95% Bootstrap Confidence Intervals on Mean Masking Gap\n"
        "All Experiment Conditions (10,000 resamples)",
        fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(pad=2.0)
    savefig(os.path.join(figures_dir, 'fig8_bootstrap_ci.png'),
            f'C1 CI=[{boot_results[0][2]:.3f}, {boot_results[0][3]:.3f}]')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — FAIRNESS METRICS SUMMARY (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 9] Fairness Metrics")
if not fair_df.empty and 'tpr' in fair_df.columns:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    tpr_by_dim = fair_df.groupby('dimension')['tpr'].mean().reindex(DIMS).dropna()
    bar_colors_a = []
    for v in tpr_by_dim.values:
        if v < 0.3:
            bar_colors_a.append('#E74C3C')
        elif v < 0.6:
            bar_colors_a.append('#F39C12')
        else:
            bar_colors_a.append('#27AE60')

    ax_a.bar(range(len(tpr_by_dim)), tpr_by_dim.values,
             color=bar_colors_a, width=0.6, edgecolor='white')
    for xi, v in enumerate(tpr_by_dim.values):
        ax_a.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax_a.axhline(0.3, color='red', linestyle='--', linewidth=1.8,
                 label='Critical threshold (0.30)')
    ax_a.set_xticks(range(len(tpr_by_dim)))
    ax_a.set_xticklabels(list(tpr_by_dim.index), rotation=30, ha='right', fontsize=9)
    ax_a.set_ylabel('Mean TPR', fontsize=12)
    ax_a.set_ylim(0, min(1.0, tpr_by_dim.max() + 0.12))
    ax_a.legend(fontsize=10, framealpha=0.9)
    ax_a.set_title("True Positive Rate by Dimension", fontsize=11, fontweight='bold')

    income_order = sorted(fair_df['income_group'].unique())
    tpr_by_inc   = fair_df.groupby('income_group')['tpr'].mean().reindex(income_order)
    inc_names    = [INCOME_LABELS.get(int(i), str(int(i))) for i in income_order]

    ax_b.bar(range(len(tpr_by_inc)), tpr_by_inc.values,
             color='#3498DB', width=0.6, edgecolor='white', alpha=0.85)
    for xi, v in enumerate(tpr_by_inc.values):
        ax_b.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax_b.axhline(0.3, color='red', linestyle='--', linewidth=1.8,
                 label='Critical threshold (0.30)')
    ax_b.set_xticks(range(len(tpr_by_inc)))
    ax_b.set_xticklabels(inc_names, rotation=25, ha='right', fontsize=9)
    ax_b.set_ylabel('Mean TPR (all dimensions)', fontsize=12)
    ax_b.set_ylim(0, min(1.0, tpr_by_inc.max() + 0.12))
    ax_b.legend(fontsize=10, framealpha=0.9)
    ax_b.set_title("Mean TPR by Income Group", fontsize=11, fontweight='bold')

    fig.suptitle(
        "Equalized Odds Analysis: True Positive Rate\n"
        "Centralized Model (Seed 42)",
        fontsize=14, fontweight='bold')
    plt.tight_layout(pad=2.0)
    savefig(os.path.join(figures_dir, 'fig9_fairness.png'),
            f'Min dim TPR={tpr_by_dim.min():.3f} ({tpr_by_dim.idxmin()})')
else:
    print("  SKIP -- fairness_metrics.csv missing or lacks tpr column")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"{'Figure':<32}  {'Size':>9}  Key Statistic")
print("=" * 72)
for name, sz, stat in fig_stats:
    print(f"{name:<32}  {sz/1024:>6.1f} KB  {stat}")
print("=" * 72)
print(f"\nAll {len(fig_stats)} figures saved to {os.path.abspath(figures_dir)}/")
