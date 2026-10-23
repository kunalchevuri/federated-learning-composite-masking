"""
Build the enriched NHANES 2015-2018 feature set for the analytic sample.

The baseline stage uses six inputs (age, gender, poverty-income ratio,
race/ethnicity, mean systolic and diastolic BP). This script adds non-leaking
predictors measured on the full adult exam sample (no fasting subsample, so
no client cell shrinks), keyed on SEQN.

Leakage rule: no variable from the questionnaire or exam component that
defines a label is used.
  Sleep      <- SLQ (SLD012)          excluded: all SLQ
  Activity   <- PAQ MET computation   excluded: all PAQ
  Sedentary  <- PAD680                excluded: all PAQ/PAD
  Mental     <- DPQ010-090 (PHQ-9)    excluded: all DPQ
  Metabolic  <- BMXBMI                excluded: all BMX (BMI, waist, weight)
  Wellbeing  <- HSD010                excluded: HSQ and HUQ010 (same item)
  Diet       <- DR1TSUGR              excluded: all DR1/DR2

Raw files: data/nhanes_raw/{COMPONENT}_{I,J}.xpt from
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{2015,2017}/DataFiles/

Output: results/nhanes_features_rich.csv (SEQN + raw added columns; scaling,
imputation and one-hot encoding happen per seed on training data only, in
revision_experiments.py).
"""
import numpy as np
import pandas as pd

RAW_DIR   = "data/nhanes_raw/"
BASE_PATH = "results/nhanes_clean.csv"
OUT_PATH  = "results/nhanes_features_rich.csv"

COMPONENTS = {
    'DEMO':   ['DMDEDUC2', 'DMDMARTL', 'DMDHHSIZ', 'DMDBORN4'],
    'BPX':    ['BPXPLS'],
    'GHB':    ['LBXGH'],
    'TCHOL':  ['LBXTC'],
    'HDL':    ['LBDHDD'],
    'SMQ':    ['SMQ020', 'SMQ040'],
    'HIQ':    ['HIQ011'],
    'HUQ':    ['HUQ030', 'HUQ051', 'HUQ071'],
    'DIQ':    ['DIQ010'],
    'BPQ':    ['BPQ020', 'BPQ080'],
    'MCQ':    ['MCQ010', 'MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F', 'MCQ220'],
    'RXQ_RX': ['RXDUSE', 'RXDCOUNT'],
}


def yes_no(s):
    """NHANES 1=yes, 2=no, 7/9=refused/don't know -> 1/0/NaN."""
    return s.map({1: 1.0, 2: 0.0})


def load_component(name, cols):
    parts = []
    for cycle in 'IJ':
        d = pd.read_sas(f"{RAW_DIR}{name}_{cycle}.xpt", format='xport')
        if name == 'RXQ_RX':
            d = d.drop_duplicates('SEQN')   # one row per prescription; person-level cols repeat
        parts.append(d[['SEQN'] + cols])
    out = pd.concat(parts)
    out['SEQN'] = out['SEQN'].astype(int)
    return out.set_index('SEQN')


def main():
    base = pd.read_csv(BASE_PATH)
    seqn = base['SEQN'].astype(int)
    raw = pd.concat([load_component(n, c) for n, c in COMPONENTS.items()], axis=1).reindex(seqn)

    f = pd.DataFrame(index=raw.index)
    # demographics / social
    f['education']      = raw['DMDEDUC2'].where(raw['DMDEDUC2'].between(1, 5))          # ordinal 1-5
    f['marital']        = raw['DMDMARTL'].map({1: 'married', 6: 'married', 2: 'prev_married',
                                               3: 'prev_married', 4: 'prev_married', 5: 'never'})
    f['household_size'] = raw['DMDHHSIZ']
    f['us_born']        = raw['DMDBORN4'].map({1: 1.0, 2: 0.0})
    # exam / labs (full exam sample, non-fasting)
    f['pulse']          = raw['BPXPLS']
    f['hba1c']          = raw['LBXGH']
    f['total_chol']     = raw['LBXTC']
    f['hdl']            = raw['LBDHDD']
    # behaviour / access / history
    smoke = pd.Series('never', index=raw.index)
    smoke[raw['SMQ020'] == 1] = 'former'
    smoke[(raw['SMQ020'] == 1) & raw['SMQ040'].isin([1, 2])] = 'current'
    smoke[~raw['SMQ020'].isin([1, 2])] = np.nan
    f['smoking']        = smoke
    f['insured']        = yes_no(raw['HIQ011'])
    f['routine_care']   = raw['HUQ030'].map({1: 1.0, 3: 1.0, 2: 0.0})
    f['care_visits']    = raw['HUQ051'].where(raw['HUQ051'].between(0, 8))              # ordinal 0-8
    f['hospitalized']   = yes_no(raw['HUQ071'])
    f['diabetes_dx']    = raw['DIQ010'].map({1: 1.0, 3: 1.0, 2: 0.0})
    f['htn_dx']         = yes_no(raw['BPQ020'])
    f['high_chol_dx']   = yes_no(raw['BPQ080'])
    f['asthma_dx']      = yes_no(raw['MCQ010'])
    f['cvd_dx']         = (raw[['MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F']] == 1).any(axis=1).astype(float)
    f.loc[raw[['MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F']].isna().all(axis=1), 'cvd_dx'] = np.nan
    f['cancer_dx']      = yes_no(raw['MCQ220'])
    rx = raw['RXDCOUNT'].copy()
    rx[raw['RXDUSE'] == 2] = 0
    f['n_prescriptions'] = rx.where(rx < 77)

    f = f.reset_index().rename(columns={'index': 'SEQN'})
    f.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {f.shape}")
    print("Missing share per feature:")
    print(f.drop(columns='SEQN').isna().mean().round(3).to_string())


if __name__ == '__main__':
    main()
