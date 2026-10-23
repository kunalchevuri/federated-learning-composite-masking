# Masking or Noise? Null-Calibrated Auditing of Per-Dimension Failures Behind Composite Wellness Scores in Federated Learning

Code, data pipeline and results for the paper of the same title, accepted at the ARRL-Health workshop at IEEE BIBM 2026.

Consumer wellness apps combine several health dimensions (sleep, activity, mood, ...) into one composite score, and the composite can look fine while one dimension fails. This repository:

- defines the **Masking Gap** (MG): the mean per-dimension AUC minus the lowest per-dimension AUC;
- derives MG's distribution under chance (random-score and label-permutation nulls), reports **MG-excess** over chance, and sets severity tiers from the null's 95th and 99th percentiles;
- gives a closed-form approximation of the chance level, which tells you how many test records per client are needed to resolve a given gap;
- measures the **composite blind spot**, Δ_d = AUC(p_d) − AUC(mean_d p), which is how much worse the composite flags dimension *d* than that dimension's own output does;
- trains nine centralized, federated and per-dimension conditions on 20 income × age clients built from NHANES 2015–2018, in a baseline stage and a full protocol.

## Data

The data come from NHANES 2015–2018 (CDC, public domain): 8,353 adults with seven binary "poor" labels.

| Dimension | Variable | Poor if |
|---|---|---|
| Sleep | SLD012 | < 7 h/night |
| Activity | GPAQ work + transport + leisure MET | < 600 MET-min/week |
| Sedentary | PAD680 | > 360 min/day sitting |
| Mental | PHQ-9 (DPQ010–090) | score ≥ 10 |
| Metabolic | BMXBMI | BMI ≥ 30 |
| Wellbeing | HSD010 | fair or poor |
| Diet | DR1TSUGR | total sugars > sample median |

The clients are the 5 poverty-income-ratio bands × 4 age bands (20 clients). The baseline inputs are age, gender, poverty-income ratio, race/ethnicity, and mean systolic and diastolic blood pressure. The enriched set (38 columns after encoding) adds non-leaking exam and questionnaire variables; see [src/build_features.py](src/build_features.py) for the leakage rule.

[results/nhanes_clean.csv](results/nhanes_clean.csv) (inputs and labels) and [results/nhanes_features_rich.csv](results/nhanes_features_rich.csv) (enriched features) are included, so the raw download is only needed to rebuild them.

## Conditions

| Code | Condition | Name in code |
|---|---|---|
| C1 | Centralized, one network with seven outputs | `centralized` |
| C2 | Seven separate centralized single-output networks | `central_separate` |
| C2-FL | Seven separate single-output FedAvg federations | `fed_separate` |
| C3 | FedAvg | `fedavg` |
| C4 | SCAFFOLD (Adam in the baseline stage, SGD in the full protocol) | `scaffold_adam` / `scaffold` |
| C5 | CS-Agg (cosine-similarity-weighted aggregation after warm-up) | `csagg` |
| C6 | Ditto | `ditto` |
| C7 | q-FedAvg | `qfedavg` |
| C8 | FedAvg with one update step per dimension on every local batch | `seqlocal` |
| C9 | FedProx (full protocol only) | `fedprox_mu*` |
| +cw, +focal | C1 / C3 with class-weighted or focal loss (full protocol only) | `*_pw`, `*_focal` |

All conditions use the same model: a 128-64 ReLU MLP with seven sigmoid heads, batch size 32, 3 local epochs, 50 FL rounds and 150 centralized epochs. The per-seed test fold (20%, stratified by client) is the same in both stages.

- **Baseline stage:** raw six inputs, Adam at lr 1e-3, final model, seeds 42/123/456.
- **Full protocol:**
  - standardized inputs
  - a 10% validation split with checkpoint selection
  - learning rates tuned on a separate seed (grid widened until every choice is interior)
  - tuned Ditto λ, q-FedAvg q and CS-Agg warm-up
  - 10 seeds
  - both feature sets
  - two extra client partitions (random IID; race/ethnicity × sex)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Everything runs on CPU. The scripts are run from the repository root.

## Reproducing the paper

```bash
# 1. data (optional: the two CSVs are already in results/)
python src/download_nhanes.py                 # raw .xpt files -> data/nhanes_raw/
python src/build_labels.py build              # -> results/nhanes_clean.csv
python src/build_features.py                  # -> results/nhanes_features_rich.csv

# 2. training (writes per-run predictions to results/revision/*.npz)
python src/revision_experiments.py baseline   # baseline stage: 9 conditions x 3 seeds
python src/revision_experiments.py tune       # learning-rate grid on the tuning seed
python src/revision_experiments.py extend     # widen the grid until every choice is interior
python src/revision_experiments.py hptune     # Ditto / q-FedAvg / CS-Agg hyperparameters
python src/revision_experiments.py run        # full protocol: all conditions x 10 seeds x 2 feature sets
python src/revision_experiments.py partition  # FedAvg under the two alternative partitions
python src/revision_experiments.py ablate     # input scaling vs. protocol, FedAvg

# 3. analysis
python src/baseline_to_csv.py                 # -> results/baseline_results.csv
python src/null_calibration.py                # baseline stage: nulls, MG-excess, power curve, Figs. 11-12
python src/analyze_revision.py                # full protocol: permutation nulls, blind spot, thresholds, fairness, Figs. 13-14
python src/contrasts_ttest.py                 # paired contrasts between conditions over seeds
python src/analyze_partitions.py              # alternative partitions
python src/shap_stability.py                  # SHAP importance across seeds
```

Set `REV_WORKERS` (training) or `SHAP_WORKERS` to control the number of parallel processes. The slowest conditions (C2, C2-FL, C6, C8) take hours on a laptop. [colab/](colab/) runs a fixed job list ([colab/jobs.json](colab/jobs.json)) on a Colab high-CPU runtime and saves each finished run to Google Drive; copy the resulting `.npz` files into `results/revision/`.

The per-run predictions and checkpoints (`results/revision/*.npz`, `*.pt`, about 85 MB) are not tracked. Every result table and the selected hyperparameters ([results/revision/tuning.csv](results/revision/tuning.csv), [results/revision/hp_tuning.csv](results/revision/hp_tuning.csv)) are.

## Where each result comes from

| Paper item | Script | Output |
|---|---|---|
| Table I (dimensions, prevalence, evaluable clients) | `build_labels.py`, `null_calibration.py` | `results/nhanes_clean.csv`, `results/evaluability.csv` |
| Table III, Fig. 11 (baseline MG vs. null) | `null_calibration.py` | `results/null_calibration_summary.csv`, `null_calibration_clients.csv`, `figures/fig11_null_calibration.png` |
| Fig. 12, Eq. 2 (null MG vs. test-set size) | `null_calibration.py` | `results/null_power_curve.csv`, `figures/fig12_power_curve.png` |
| Table IV (population AUCs) | `null_calibration.py`, `analyze_revision.py` | `results/population_dim_auc.csv`, `results/revision_table_population.csv` |
| Table V, Fig. 13 (full-protocol MG, per-dimension AUC) | `analyze_revision.py` | `results/revision_table_client_mg.csv`, `revision_table_dims.csv`, `figures/fig13_dimension_auc_by_regime.png` |
| Fig. 14 (composite blind spot) | `analyze_revision.py` | `results/revision_label_masking.csv`, `figures/fig14_composite_blind_spot.png` |
| Contrasts between conditions | `contrasts_ttest.py`, `null_calibration.py` | `results/revision_contrasts_ttest.csv`, `results/pairwise_tests.csv`, `results/mixed_model.txt` |
| Thresholds, calibration, fairness | `analyze_revision.py` | `results/revision_dims.csv`, `results/revision_table_fairness.csv`, `results/revision_groups.csv` |
| Alternative partitions | `analyze_partitions.py` | `results/revision_partitions.csv` |
| SHAP stability | `shap_stability.py` | `results/revision_shap_summary.csv` |

## Evaluation rule

A per-client, per-dimension AUC is computed only when the client's test fold has at least 5 positive and 5 negative cases. Other cells are treated as missing, not as 0.5.

## Repository layout

```
src/        data pipeline, experiments and analysis
results/    analytic sample, result tables, tuning results
figures/    paper figures (Figs. 11-14)
paper/      LaTeX source of the paper (paper/workshop/main.tex)
colab/      runner for the slow full-protocol jobs on Google Colab
```

## Citation

If you use this code, please cite the paper: *Masking or Noise? Null-Calibrated Auditing of Per-Dimension Failures Behind Composite Wellness Scores in Federated Learning*, ARRL-Health Workshop, IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2026.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). NHANES data are public domain (CDC/NCHS).
