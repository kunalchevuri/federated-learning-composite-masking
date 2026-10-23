"""
Experiments for the composite-masking study: a baseline stage and a full protocol.

Shared by every run: MultiOutputMLP (128-64 ReLU backbone, seven sigmoid heads),
Adam, batch 32, 3 local epochs, 50 FL rounds, 150 centralized epochs, 20 income x
age clients, and the same per-seed test fold in every stage
(train_test_split(test_size=0.2, random_state=seed, stratify=client_id)).

Baseline stage ('baseline'): six raw inputs, lr 1e-3 for every condition, final
model evaluated, seeds 42/123/456.

Full protocol:
  * validation split: 12.5% of the training fold (10% overall), stratified by
    client; the checkpoint (epoch / round) with the best pooled validation
    macro-AUC is used for testing
  * learning rate tuned on a separate tuning seed (2026) over a grid, selected
    on validation only; the grid is extended outward whenever the selected
    value lies at its edge
  * method hyperparameters (Ditto lambda, q-FedAvg q, CS-Agg warm-up) tuned on
    the tuning seed at each method's selected learning rate
  * 10 evaluation seeds
  * preprocessing: continuous inputs standardized and race one-hot encoded,
    fitted on the training fold
  * enriched feature set (build_features.py)
  * class-weighted BCE and focal-loss baselines, and FedProx
  * test and validation predictions are saved, enabling label-permutation
    nulls, AUPRC / Brier / ECE and validation-tuned thresholds
  * alternative client partitions (random IID; race/ethnicity x sex)

FL methods: SCAFFOLD uses plain local SGD in the full protocol, for which its
control-variate update is defined (the baseline stage uses Adam for every
condition, 'scaffold_adam'); CS-Agg uses plain FedAvg for its warm-up rounds,
then weights clients by their shifted mean cosine similarity to the other
clients' updates; Ditto trains a personal model per client toward the new
global model each round and predicts each client's records with that client's
personal model; q-FedAvg weights clients by their post-training composite
loss; C8 takes one Adam step per dimension on every local batch. FedProx adds
(mu / 2) ||w - w_global||^2 to the local loss, with mu tuned with the
learning rate.

Usage (in this order; set REV_WORKERS to limit parallel workers):
  python src/revision_experiments.py baseline    # baseline stage
  python src/revision_experiments.py tune        # lr grid on the tuning seed
  python src/revision_experiments.py extend      # widen the grid until every choice is interior
  python src/revision_experiments.py hptune      # method hyperparameters on the tuning seed
  python src/revision_experiments.py run         # all full-protocol evaluation jobs
  python src/revision_experiments.py partition   # FedAvg under alternative partitions
  python src/revision_experiments.py ablate      # input scaling vs protocol, FedAvg
"""
import copy
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE_PATH = "results/nhanes_clean.csv"
RICH_PATH = "results/nhanes_features_rich.csv"
OUT_DIR   = "results/revision/"
os.makedirs(OUT_DIR, exist_ok=True)

N_ROUNDS   = 50
N_EPOCHS   = 3        # local epochs per FL round
C_EPOCHS   = 150      # centralized epochs
BATCH_SIZE = 32
N_CLIENTS  = 20
N_DIMS     = 7
FOCAL_GAMMA = 2.0
POS_WEIGHT_CAP = 20.0

EVAL_SEEDS = [42, 123, 456, 0, 1, 2, 3, 4, 5, 6]
TUNE_SEED  = 2026
LR_GRID    = [3e-3, 1e-3, 3e-4, 1e-4]
# SCAFFOLD is defined for local SGD (Karimireddy et al.); its step sizes live on another scale
LR_GRIDS   = {'scaffold': [3e-1, 1e-1, 3e-2, 1e-2]}
# if the best value is at an edge of what was tried, the grid is extended one
# step outward along this ladder until the selected value is interior
LR_LADDER  = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
LR_LADDERS = {'scaffold': [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3]}
N_WORKERS  = int(os.environ.get("REV_WORKERS", 11))
CSAGG_WARMUP = 10
DITTO_LAMBDA = 0.1
QFED_Q       = 1
# method hyperparameters tuned on TUNE_SEED at each method's selected lr ('hptune');
# the defaults above are the middle values
HP_GRID = {'ditto': ('_lam', [0.01, 0.1, 1.0]), 'qfedavg': ('_q', [0.1, 1.0, 5.0]),
           'csagg': ('_w', [5, 10, 20])}
PARTITIONS   = ['random', 'race_sex']

FEATURE_COLS = ["RIDAGEYR", "RIAGENDR", "INDFMPIR", "RIDRETH3", "avg_sys", "avg_dia"]
LABEL_COLS   = ["sleep_poor", "activity_poor", "sedentary_poor",
                "mental_poor", "metabolic_poor", "wellbeing_poor", "diet_poor"]
DIM_NAMES    = ["Sleep", "Activity", "Sedentary", "Mental", "Metabolic", "Wellbeing", "Diet"]
INCOME_BINS  = [0, 1.0, 2.0, 3.0, 4.0, 99]
AGE_BINS     = [18, 36, 51, 65, 120]

# condition -> (training regime, loss)
CONDITIONS = {
    'centralized':       ('central', 'bce'),
    'centralized_pw':    ('central', 'pw'),
    'centralized_focal': ('central', 'focal'),
    'fedavg':            ('fedavg',  'bce'),
    'fedavg_pw':         ('fedavg',  'pw'),
    'fedavg_focal':      ('fedavg',  'focal'),
    'fed_separate':      ('fedsep',  'bce'),
    'central_separate':  ('censep',  'bce'),     # C2
    'scaffold':          ('scaffold', 'bce'),    # C4
    'csagg':             ('csagg',   'bce'),     # C5
    'ditto':             ('ditto',   'bce'),     # C6
    'qfedavg':           ('qfedavg', 'bce'),     # C7
    'seqlocal':          ('seqlocal', 'bce'),    # C8
    'fedprox_mu0.001':   ('fedprox', 'bce'),
    'fedprox_mu0.01':    ('fedprox', 'bce'),
    'fedprox_mu0.1':     ('fedprox', 'bce'),
}
# conditions whose lr is inherited rather than tuned (7x cost)
INHERIT_LR = {'fed_separate': 'fedavg', 'central_separate': 'centralized'}
FEDPROX = [c for c in CONDITIONS if c.startswith('fedprox')]
HP_VARIANTS = {f'{m}{key}{v:g}': CONDITIONS[m] for m, (key, vals) in HP_GRID.items() for v in vals}
# baseline stage: raw inputs, Adam at lr 1e-3 for every condition (SCAFFOLD included),
# final model evaluated, seeds 42/123/456
BASELINE_ONLY  = {'scaffold_adam': ('scaffold', 'bce')}
BASELINE_CONDS = ['centralized', 'central_separate', 'fed_separate', 'fedavg', 'scaffold_adam',
                  'csagg', 'ditto', 'qfedavg', 'seqlocal']
BASELINE_SEEDS = [42, 123, 456]


def base_cond(c):
    for m, (key, _) in HP_GRID.items():
        if c.startswith(m + key):
            return m
    return c


def hp_value(cond, method, default):
    key = HP_GRID[method][0]
    return float(cond.split(key)[1]) if cond.startswith(method + key) else default
FEATURE_SETS = ['orig6', 'rich']


# ── data ───────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(BASE_PATH)
    df['income_group'] = pd.cut(df['INDFMPIR'], bins=INCOME_BINS, labels=[0, 1, 2, 3, 4],
                                right=False).astype(int)
    df['age_group'] = pd.cut(df['RIDAGEYR'], bins=AGE_BINS, labels=[0, 1, 2, 3],
                             right=False).astype(int)
    df['client_id'] = df['income_group'] * 4 + df['age_group']
    rich = pd.read_csv(RICH_PATH)
    df['SEQN'] = df['SEQN'].astype(int)
    return df.merge(rich, on='SEQN', how='left', validate='one_to_one')


def partition_ids(df, partition):
    """Client assignment. The test fold never depends on it (split() always
    stratifies on the income x age clients), so partitions share test data."""
    if partition == 'income_age':
        return df['client_id'].values
    if partition == 'random':
        # fixed IID assignment of records to 20 equal-sized clients (same for every seed)
        rng = np.random.default_rng(20260924)
        return rng.permutation(np.arange(len(df)) % N_CLIENTS)
    if partition == 'race_sex':
        codes = {r: i for i, r in enumerate(sorted(df['RIDRETH3'].astype(int).unique()))}
        return (df['RIDRETH3'].astype(int).map(codes) * 2 + (df['RIAGENDR'] == 2)).values.astype(int)
    raise ValueError(partition)


def split(df, seed, use_val):
    """Same test fold for every stage and condition; validation carved from train."""
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=seed,
                                           stratify=df['client_id'])
    idx_val = pd.Index([])
    if use_val:
        idx_train, idx_val = train_test_split(idx_train, test_size=0.125, random_state=seed,
                                              stratify=df.loc[idx_train, 'client_id'])
    return idx_train, idx_val, idx_test


def design_matrix(df, fs, idx_train):
    """Feature matrix for feature set `fs`; all statistics fitted on idx_train."""
    if fs == 'orig6_raw':
        return df[FEATURE_COLS].values.astype(np.float32)
    X = pd.DataFrame(index=df.index)
    X['age']    = df['RIDAGEYR']
    X['female'] = (df['RIAGENDR'] == 2).astype(float)
    X['pir']    = df['INDFMPIR']
    X['sys']    = df['avg_sys']
    X['dia']    = df['avg_dia']
    X = X.join(pd.get_dummies(df['RIDRETH3'].astype(int), prefix='race', dtype=float))
    if fs == 'rich':
        num = ['education', 'household_size', 'us_born', 'pulse', 'hba1c', 'total_chol', 'hdl',
               'insured', 'routine_care', 'care_visits', 'hospitalized', 'diabetes_dx', 'htn_dx',
               'high_chol_dx', 'asthma_dx', 'cvd_dx', 'cancer_dx', 'n_prescriptions']
        X = X.join(df[num].astype(float))
        X = X.join(pd.get_dummies(df['marital'], prefix='marital', dtype=float))
        X = X.join(pd.get_dummies(df['smoking'], prefix='smoke', dtype=float))
        for block, cols in {'lab_missing': ['hba1c', 'total_chol', 'hdl'],
                            'mcq_missing': ['cvd_dx', 'cancer_dx'],
                            'edu_missing': ['education']}.items():
            X[block] = df[cols].isna().any(axis=1).astype(float)
    med = X.loc[idx_train].median()
    X = X.fillna(med)
    mu, sd = X.loc[idx_train].mean(), X.loc[idx_train].std().replace(0, 1)
    return ((X - mu) / sd).values.astype(np.float32)


# ── models / losses ────────────────────────────────────────────────────────────
class MultiOutputMLP(nn.Module):
    def __init__(self, n_in, n_out=N_DIMS):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_out)])

    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([torch.sigmoid(head(h)) for head in self.heads], dim=1)


def loss_fn(p, y, kind, pos_w):
    """Per-dimension mean, summed across dimensions (the original composite objective)."""
    p = p.clamp(1e-7, 1 - 1e-7)
    if kind == 'pw':
        l = -(pos_w * y * torch.log(p) + (1 - y) * torch.log(1 - p))
    else:
        l = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        if kind == 'focal':
            pt = y * p + (1 - y) * (1 - p)
            l = (1 - pt) ** FOCAL_GAMMA * l
    return l.mean(dim=0).sum()


def pos_weights(y):
    pos = y.sum(0)
    return ((len(y) - pos) / pos.clamp(min=1)).clamp(max=POS_WEIGHT_CAP)


def train_epochs(model, X, y, epochs, lr, kind, opt=None, prox=None, per_dim=False, cv=None,
                 sgd=True):
    """prox = (weight, anchor params): adds (weight / 2) ||w - anchor||^2 (FedProx, Ditto).
    per_dim: one gradient step per dimension on every batch (C8).
    cv = (c_local, c_global): SCAFFOLD correction added to each gradient, with plain
    local SGD as in Karimireddy et al. (sgd=False: Adam steps, baseline stage only)."""
    if opt is None:
        opt = (torch.optim.SGD if cv is not None and sgd else torch.optim.Adam)(model.parameters(), lr=lr)
    pw = pos_weights(y)
    model.train()
    n_steps = 0
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            for d in (range(y.shape[1]) if per_dim else [None]):
                opt.zero_grad()
                p = model(X[b])
                loss = (loss_fn(p, y[b], kind, pw) if d is None else
                        loss_fn(p[:, d:d + 1], y[b][:, d:d + 1], kind, pw[d:d + 1]))
                if prox is not None:
                    mu, anchor = prox
                    loss = loss + (mu / 2) * sum(((w - a) ** 2).sum()
                                                 for w, a in zip(model.parameters(), anchor))
                loss.backward()
                if cv is not None:
                    with torch.no_grad():
                        for w, ci, cg in zip(model.parameters(), *cv):
                            w.grad += (cg - ci)
                opt.step()
                n_steps += 1
    return opt if cv is None else n_steps


def fedavg(global_model, local_models, sizes):
    total = sum(sizes)
    weighted_average(global_model, local_models, [n / total for n in sizes])


def weighted_average(global_model, local_models, weights):
    sd = global_model.state_dict()
    for k in sd:
        sd[k] = sum(w * m.state_dict()[k] for m, w in zip(local_models, weights))
    global_model.load_state_dict(sd)


def flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def csagg_weights(pre, local_models):
    """Mean cosine similarity of each client's update to the others', shifted and normalized."""
    D = torch.stack([flat(m) - pre for m in local_models])
    D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-12)
    S = (D @ D.T).numpy()
    n = len(local_models)
    trust = (S.sum(1) - np.diag(S)) / (n - 1)
    trust = trust - trust.min() + 1e-8
    return trust / trust.sum()


def composite_loss(model, X, y):
    """Sum over dimensions of the mean BCE on the client's training data (q-FedAvg weights)."""
    with torch.no_grad():
        p = model(X).clamp(1e-7, 1 - 1e-7)
        return float((-(y * torch.log(p) + (1 - y) * torch.log(1 - p))).mean(0).sum())


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, d], p[:, d]) for d in range(y.shape[1])]))


# ── one job ────────────────────────────────────────────────────────────────────
def run_job(job):
    fs, cond, seed, lr, use_val, tag = job[:6]
    partition = job[6] if len(job) > 6 else 'income_age'
    out_path = f"{OUT_DIR}{tag}__{fs}__{cond}__s{seed}__lr{lr:g}.npz"
    if os.path.exists(out_path):
        return out_path, 'cached', 0.0
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    df = load_data()
    idx_tr, idx_va, idx_te = split(df, seed, use_val)
    Xall = design_matrix(df, fs, idx_tr)
    Yall = df[LABEL_COLS].values.astype(np.float32)
    cid = partition_ids(df, partition)
    T = lambda idx: (torch.tensor(Xall[idx]), torch.tensor(Yall[idx]))
    X_te, y_te = T(idx_te)
    X_va, y_va = T(idx_va) if use_val else (None, None)
    cid_te = cid[idx_te]
    cid_va = cid[idx_va] if use_val else None
    clients = []
    for c in range(int(cid.max()) + 1):
        idx = idx_tr[cid[idx_tr] == c]
        clients.append(T(idx))
    regime, kind = {**CONDITIONS, **HP_VARIANTS, **BASELINE_ONLY}[cond]
    ditto_lam = hp_value(cond, 'ditto', DITTO_LAMBDA)
    qfed_q    = hp_value(cond, 'qfedavg', QFED_Q)
    warmup    = int(hp_value(cond, 'csagg', CSAGG_WARMUP))
    n_in = Xall.shape[1]

    curve, best = [], {'score': -np.inf, 'step': -1, 'p_te': None, 'p_va': None}

    def checkpoint(step, model_or_preds, save_state=True):
        """model_or_preds(X, client_ids) -> predictions."""
        p_te = model_or_preds(X_te, cid_te)
        if use_val:
            p_va = model_or_preds(X_va, cid_va)
            score = macro_auc(y_va.numpy(), p_va)
            curve.append((step, score))
            if score > best['score']:
                best.update(score=score, step=step, p_te=p_te, p_va=p_va,
                            state=copy.deepcopy(model.state_dict()) if save_state else None)
        return p_te

    if regime == 'central':
        X_tr = torch.cat([c[0] for c in clients]); y_tr = torch.cat([c[1] for c in clients])
        model = MultiOutputMLP(n_in)
        opt = None
        for ep in range(1, C_EPOCHS + 1):
            opt = train_epochs(model, X_tr, y_tr, 1, lr, kind, opt)
            if use_val and (ep % 5 == 0 or ep == 1):
                checkpoint(ep, lambda X, c: predict(model, X))
        p_te_final = predict(model, X_te)
        p_va_final = predict(model, X_va) if use_val else None

    elif regime in ('fedavg', 'fedprox', 'seqlocal', 'scaffold', 'csagg', 'qfedavg', 'ditto'):
        model = MultiOutputMLP(n_in)
        sizes = [len(c[0]) for c in clients]
        mu = float(cond.split('_mu')[1]) if regime == 'fedprox' else None
        if regime == 'scaffold':
            c_global = [torch.zeros_like(p) for p in model.parameters()]
            c_locals = [[torch.zeros_like(p) for p in model.parameters()] for _ in clients]
        if regime == 'ditto':
            personal = [copy.deepcopy(model) for _ in clients]

        def predict_personal(X, c):
            out = np.zeros((len(X), N_DIMS), np.float32)
            for k in np.unique(c):
                out[c == k] = predict(personal[k], X[torch.from_numpy(c == k)])
            return out

        for rnd in range(N_ROUNDS):
            pre = flat(model)
            anchor = [p.detach().clone() for p in model.parameters()]
            locals_, losses, new_c = [], [], []
            for k, (Xc, yc) in enumerate(clients):
                m = copy.deepcopy(model)
                if regime == 'scaffold':
                    n_steps = train_epochs(m, Xc, yc, N_EPOCHS, lr, kind, cv=(c_locals[k], c_global),
                                           sgd=(cond != 'scaffold_adam'))
                    new_c.append([(ci - cg + (w0 - w.data) / (max(n_steps, 1) * lr)).clone()
                                  for ci, cg, w0, w in zip(c_locals[k], c_global, anchor, m.parameters())])
                else:
                    train_epochs(m, Xc, yc, N_EPOCHS, lr, kind,
                                 prox=(mu, anchor) if regime == 'fedprox' else None,
                                 per_dim=(regime == 'seqlocal'))
                if regime == 'qfedavg':
                    losses.append(composite_loss(m, Xc, yc))
                locals_.append(m)
            if regime == 'csagg' and rnd >= warmup:
                weighted_average(model, locals_, csagg_weights(pre, locals_))
            elif regime == 'qfedavg':
                w = np.array(losses) ** qfed_q
                weighted_average(model, locals_, w / w.sum())
            else:
                fedavg(model, locals_, sizes)
            if regime == 'scaffold':
                for i in range(len(c_global)):
                    c_global[i] += sum(new_c[k][i] - c_locals[k][i] for k in range(len(clients))) / len(clients)
                c_locals = new_c
            if regime == 'ditto':
                g = [p.detach().clone() for p in model.parameters()]
                for k, (Xc, yc) in enumerate(clients):
                    train_epochs(personal[k], Xc, yc, N_EPOCHS, lr, kind, prox=(ditto_lam, g))
            if use_val:
                if regime == 'ditto':
                    checkpoint(rnd, predict_personal, save_state=False)
                else:
                    checkpoint(rnd, lambda X, c: predict(model, X))
        if regime == 'ditto':
            p_te_final = predict_personal(X_te, cid_te)
            p_va_final = predict_personal(X_va, cid_va) if use_val else None
        else:
            p_te_final = predict(model, X_te)
            p_va_final = predict(model, X_va) if use_val else None

    elif regime == 'censep':
        # seven independent single-output centralized classifiers (C2); each selects its own epoch
        X_tr = torch.cat([c[0] for c in clients]); y_tr = torch.cat([c[1] for c in clients])
        p_te_final = np.zeros((len(X_te), N_DIMS), np.float32)
        p_te_best  = np.zeros_like(p_te_final)
        p_va_final = np.zeros((len(X_va), N_DIMS), np.float32) if use_val else None
        p_va_best  = np.zeros_like(p_va_final) if use_val else None
        best_steps = []
        for d in range(N_DIMS):
            torch.manual_seed(seed + d)
            model = MultiOutputMLP(n_in, n_out=1)
            bd = {'score': -np.inf, 'step': -1}
            opt = None
            for ep in range(1, C_EPOCHS + 1):
                opt = train_epochs(model, X_tr, y_tr[:, d:d + 1], 1, lr, kind, opt)
                if use_val and (ep % 5 == 0 or ep == 1):
                    pv = predict(model, X_va)[:, 0]
                    s = roc_auc_score(y_va.numpy()[:, d], pv)
                    if s > bd['score']:
                        bd.update(score=s, step=ep)
                        p_va_best[:, d] = pv
                        p_te_best[:, d] = predict(model, X_te)[:, 0]
            p_te_final[:, d] = predict(model, X_te)[:, 0]
            if use_val:
                p_va_final[:, d] = predict(model, X_va)[:, 0]
            best_steps.append(bd['step'])
        if use_val:
            best.update(score=macro_auc(y_va.numpy(), p_va_best), step=best_steps,
                        p_te=p_te_best, p_va=p_va_best)

    elif regime == 'fedsep':
        # seven independent single-output federations; each selects its own best round
        sizes = [len(c[0]) for c in clients]
        p_te_final = np.zeros((len(X_te), N_DIMS), np.float32)
        p_te_best  = np.zeros_like(p_te_final)
        p_va_final = np.zeros((len(X_va), N_DIMS), np.float32) if use_val else None
        p_va_best  = np.zeros_like(p_va_final) if use_val else None
        best_steps = []
        for d in range(N_DIMS):
            torch.manual_seed(seed + d)
            model = MultiOutputMLP(n_in, n_out=1)
            bd = {'score': -np.inf, 'step': -1}
            for rnd in range(N_ROUNDS):
                locals_ = []
                for Xc, yc in clients:
                    m = copy.deepcopy(model)
                    train_epochs(m, Xc, yc[:, d:d + 1], N_EPOCHS, lr, kind)
                    locals_.append(m)
                fedavg(model, locals_, sizes)
                if use_val:
                    pv = predict(model, X_va)[:, 0]
                    s = roc_auc_score(y_va.numpy()[:, d], pv)
                    if s > bd['score']:
                        bd.update(score=s, step=rnd)
                        p_va_best[:, d] = pv
                        p_te_best[:, d] = predict(model, X_te)[:, 0]
            p_te_final[:, d] = predict(model, X_te)[:, 0]
            if use_val:
                p_va_final[:, d] = predict(model, X_va)[:, 0]
            best_steps.append(bd['step'])
        if use_val:
            best.update(score=macro_auc(y_va.numpy(), p_va_best), step=best_steps,
                        p_te=p_te_best, p_va=p_va_best)

    if best.get('state') is not None:
        torch.save(best['state'], out_path.replace('.npz', '.pt'))
    np.savez_compressed(
        out_path,
        seqn_test=df['SEQN'].values[idx_te], client_test=cid[idx_te], y_test=Yall[idx_te],
        p_test_final=p_te_final,
        p_test_best=best['p_te'] if use_val else p_te_final,
        seqn_val=df['SEQN'].values[idx_va] if use_val else np.array([]),
        y_val=Yall[idx_va] if use_val else np.array([]),
        p_val_best=best['p_va'] if use_val else np.array([]),
        p_val_final=p_va_final if use_val else np.array([]),
        val_score_best=best['score'], best_step=np.array(best['step']),
        curve=np.array(curve), n_features=n_in, lr=lr, seed=seed,
    )
    return out_path, 'done', time.time() - t0


def run_pool(jobs):
    print(f"{len(jobs)} jobs on {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for path, status, secs in pool.imap_unordered(run_job, jobs):
            print(f"[{time.time() - t0:7.0f}s] {status:6s} {secs:6.0f}s  {os.path.basename(path)}",
                  flush=True)


def tuned_lrs():
    rows = []
    for f in os.listdir(OUT_DIR):
        if f.startswith('tune__') and f.endswith('.npz'):
            _, fs, cond, _, lr = f[:-4].split('__')
            rows.append((fs, cond, float(lr[2:]), float(np.load(OUT_DIR + f)['val_score_best'])))
    t = pd.DataFrame(rows, columns=['fs', 'cond', 'lr', 'val_macro_auc'])
    t.to_csv(OUT_DIR + 'tuning.csv', index=False)
    best = t.loc[t.groupby(['fs', 'cond'])['val_macro_auc'].idxmax()]
    return {(r.fs, r.cond): r.lr for r in best.itertuples()}


def hp_tuning():
    """(fs, method) -> best hyperparameter variant on the tuning seed (hptune runs)."""
    rows = []
    for f in os.listdir(OUT_DIR):
        if f.startswith('hptune__') and f.endswith('.npz'):
            _, fs, cond, _, lr = f[:-4].split('__')
            rows.append((fs, base_cond(cond), cond, float(np.load(OUT_DIR + f)['val_score_best'])))
    t = pd.DataFrame(rows, columns=['fs', 'method', 'variant', 'val_macro_auc'])
    if t.empty:
        return {}
    t.to_csv(OUT_DIR + 'hp_tuning.csv', index=False)
    best = t.loc[t.groupby(['fs', 'method'])['val_macro_auc'].idxmax()]
    return {(r.fs, r.method): r.variant for r in best.itertuples()}


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'tune':
        # the separate-classifier conditions inherit their joint model's lr (7x cost)
        run_pool([(fs, c, TUNE_SEED, lr, True, 'tune')
                  for fs in FEATURE_SETS for c in CONDITIONS if c not in INHERIT_LR
                  for lr in LR_GRIDS.get(c, LR_GRID)])
        print(tuned_lrs())
    elif mode == 'extend':
        # repeat until every selected lr has a tried neighbour on both sides
        while True:
            tuned_lrs()
            t = pd.read_csv(OUT_DIR + 'tuning.csv')
            jobs = []
            for (fs, c), g in t.groupby(['fs', 'cond']):
                ladder = LR_LADDERS.get(c, LR_LADDER)
                tried = sorted(g['lr'], reverse=True)
                best = g.loc[g['val_macro_auc'].idxmax(), 'lr']
                i = min(range(len(ladder)), key=lambda k: abs(np.log(ladder[k] / best)))
                for edge, step in [(tried[0], -1), (tried[-1], +1)]:
                    if np.isclose(best, edge) and 0 <= i + step < len(ladder):
                        jobs.append((fs, c, TUNE_SEED, ladder[i + step], True, 'tune'))
            if not jobs:
                break
            run_pool(jobs)
        print(tuned_lrs())
    elif mode == 'run':
        lrs = tuned_lrs()
        # archive eval runs made with a learning rate that is no longer selected
        os.makedirs(OUT_DIR + 'superseded', exist_ok=True)
        for f in os.listdir(OUT_DIR):
            if f.startswith('eval__') and f.endswith(('.npz', '.pt')):
                _, fs, c, _, lr = f.rsplit('.', 1)[0].split('__')
                if c in FEDPROX:
                    continue
                if not np.isclose(float(lr[2:]), lrs[(fs, INHERIT_LR.get(base_cond(c), base_cond(c)))]):
                    os.replace(OUT_DIR + f, OUT_DIR + 'superseded/' + f)
                    print('superseded:', f)
        t = pd.read_csv(OUT_DIR + 'tuning.csv')
        jobs = []
        for fs in FEATURE_SETS:
            # FedProx: evaluate only the mu with the best tuning score
            fp = t[(t.fs == fs) & t.cond.isin(FEDPROX)]
            best_fp = fp.loc[fp['val_macro_auc'].idxmax(), 'cond']
            hp = hp_tuning()
            for c in CONDITIONS:
                if c in FEDPROX and c != best_fp:
                    continue
                lr = lrs[(fs, INHERIT_LR.get(c, c))]
                name = hp.get((fs, c), c)
                jobs += [(fs, name, s, lr, True, 'eval') for s in EVAL_SEEDS]
        # slow jobs first so the pool drains evenly
        slow = ['fed_separate', 'central_separate', 'seqlocal', 'ditto']
        if os.environ.get('REV_ORDER') == 'fast':
            # core and fast conditions first, so a partial run already covers them
            prio = ['centralized', 'fedavg', 'scaffold', 'fedprox', 'csagg', 'qfedavg', 'fedavg_pw',
                    'fedavg_focal', 'centralized_pw', 'centralized_focal', 'ditto', 'seqlocal',
                    'fed_separate', 'central_separate']
            base = lambda c: 'fedprox' if c.startswith('fedprox') else base_cond(c)
            jobs.sort(key=lambda j: (prio.index(base(j[1])), j[2]))
        else:
            jobs.sort(key=lambda j: slow.index(j[1]) if j[1] in slow else len(slow))
        run_pool(jobs)
    elif mode == 'hptune':
        lrs = tuned_lrs()
        run_pool([(fs, v, TUNE_SEED, lrs[(fs, m)], True, 'hptune')
                  for fs in FEATURE_SETS for v, (reg, _) in HP_VARIANTS.items()
                  for m in [v.split('_')[0]]])
    elif mode == 'baseline':
        run_pool([('orig6_raw', c, s, 1e-3, False, 'baseline')
                  for c in BASELINE_CONDS for s in BASELINE_SEEDS])
    elif mode == 'ablate':
        # which part of the full protocol removes the baseline's federated failure?
        lrs = tuned_lrs()
        run_pool([('orig6', 'fedavg', s, 1e-3, False, 'ablate_scaling_only') for s in BASELINE_SEEDS] +
                 [('orig6_raw', 'fedavg', s, lrs[('orig6', 'fedavg')], True, 'ablate_protocol_only')
                  for s in BASELINE_SEEDS])
    elif mode == 'partition':
        lrs = tuned_lrs()
        run_pool([(fs, 'fedavg', s, lrs[(fs, 'fedavg')], True, f'part-{p}', p)
                  for p in PARTITIONS for fs in FEATURE_SETS for s in EVAL_SEEDS])
