"""
Rebuild the analytic sample (six base inputs + seven binary "poor" labels) from raw
NHANES files, for any set of cycles.

  python src/build_labels.py build     2015-2018 (I, J): write results/nhanes_clean.csv
  python src/build_labels.py check     2015-2018: compare with results/nhanes_clean.csv
  python src/build_labels.py 2011      2011-2014 (G, H): write results/nhanes_clean_2011.csv

Label definitions (Table I of the paper):
  Sleep      usual sleep < 7 h/night    SLD012 (2015-18), SLD010H (2011-14)
  Activity   < 600 MET-min/week         PAQ: 8 x vigorous + 4 x moderate/transport minutes
  Sedentary  > 360 min/day sitting      PAD680
  Mental     PHQ-9 total >= 10          DPQ010-DPQ090
  Metabolic  BMI >= 30                  BMXBMI
  Wellbeing  fair/poor general health   HSD010 in {4, 5}
  Diet       total sugars > median      DR1TSUGR (median of the analytic sample)
"""
import sys
import numpy as np
import pandas as pd

RAW = "data/nhanes_raw/"
YEARS = {'G': 2011, 'H': 2013, 'I': 2015, 'J': 2017}


def read(name, cycles):
    out = []
    for c in cycles:
        d = pd.read_sas(f"{RAW}{name}_{c}.xpt", format='xport')
        d['SEQN'] = d['SEQN'].astype(int)
        num = d.select_dtypes('number').columns
        d[num] = d[num].mask(d[num].abs() < 1e-10, 0.0)   # xport stores 0 as ~5e-79
        out.append(d)
    return pd.concat(out, ignore_index=True).drop_duplicates('SEQN').set_index('SEQN')


def valid(s, lo, hi):
    return s.where(s.between(lo, hi))


def met_minutes(paq):
    """MET-min/week from the GPAQ items: days/week x minutes/day x MET (8 vigorous, 4 otherwise)."""
    parts = [  # (yes item, days, minutes, MET)
        ('PAQ605', 'PAQ610', 'PAD615', 8),   # vigorous work
        ('PAQ620', 'PAQ625', 'PAD630', 4),   # moderate work
        ('PAQ635', 'PAQ640', 'PAD645', 4),   # walk/bicycle for transport
        ('PAQ650', 'PAQ655', 'PAD660', 8),   # vigorous recreation
        ('PAQ665', 'PAQ670', 'PAD675', 4),   # moderate recreation
    ]
    total = pd.Series(0.0, index=paq.index)
    ok = pd.Series(True, index=paq.index)
    for yes, days, mins, met in parts:
        y = paq[yes]
        d = valid(paq[days], 1, 7)
        m = valid(paq[mins], 1, 1440)
        contrib = (d * m * met).where(y == 1, 0.0)
        ok &= y.isin([1, 2]) & ~((y == 1) & contrib.isna())
        total += contrib.fillna(0.0)
    return total.where(ok)


def build(cycles):
    demo = read('DEMO', cycles)
    bpx = read('BPX', cycles)
    slq = read('SLQ', cycles)
    paq = read('PAQ', cycles)
    dpq = read('DPQ', cycles)
    bmx = read('BMX', cycles)
    hsq = read('HSQ', cycles)
    dr1 = read('DR1TOT', cycles)

    df = pd.DataFrame(index=demo.index)
    df['RIDAGEYR'] = demo['RIDAGEYR']
    df['RIAGENDR'] = demo['RIAGENDR']
    df['INDFMPIR'] = demo['INDFMPIR']
    df['RIDRETH3'] = demo['RIDRETH3']
    sys_cols = [c for c in ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4'] if c in bpx]
    dia_cols = [c for c in ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4'] if c in bpx]
    df['avg_sys'] = bpx[sys_cols].mean(axis=1).reindex(df.index)
    df['avg_dia'] = bpx[dia_cols].mean(axis=1).reindex(df.index)

    sleep_h = slq['SLD012'] if 'SLD012' in slq else slq['SLD010H']
    sleep_h = valid(sleep_h, 1, 24).reindex(df.index)
    met = met_minutes(paq).reindex(df.index)
    sit = valid(paq['PAD680'], 0, 1440).reindex(df.index)
    items = [f'DPQ0{i}0' for i in range(1, 10)]
    phq = dpq[items].where(dpq[items].isin([0, 1, 2, 3]))
    phq = phq.sum(axis=1, min_count=9).reindex(df.index)
    bmi = bmx['BMXBMI'].reindex(df.index)
    gh = hsq['HSD010'].where(hsq['HSD010'].between(1, 5)).reindex(df.index)
    sugar = dr1['DR1TSUGR'].reindex(df.index)
    if 'DR1DRSTZ' in dr1:
        sugar = sugar.where(dr1['DR1DRSTZ'].reindex(df.index) == 1)

    raw = pd.DataFrame({'sleep': sleep_h, 'met': met, 'sit': sit, 'phq': phq,
                        'bmi': bmi, 'gh': gh, 'sugar': sugar})
    keep = df['RIDAGEYR'].between(18, 80) & df.notna().all(axis=1) & raw.notna().all(axis=1)
    df, raw = df[keep].copy(), raw[keep]
    df['sleep_poor'] = (raw['sleep'] < 7).astype(float)
    df['activity_poor'] = (raw['met'] < 600).astype(float)
    df['sedentary_poor'] = (raw['sit'] > 360).astype(float)
    df['mental_poor'] = (raw['phq'] >= 10).astype(float)
    df['metabolic_poor'] = (raw['bmi'] >= 30).astype(float)
    df['wellbeing_poor'] = (raw['gh'] >= 4).astype(float)
    df['diet_poor'] = (raw['sugar'] > raw['sugar'].median()).astype(float)
    return df.reset_index()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if mode == 'build':
        out = build('IJ')
        out.to_csv('results/nhanes_clean.csv', index=False)
        print(f"wrote results/nhanes_clean.csv {out.shape}")
        print(out.iloc[:, 7:].mean().round(3).to_string())
    elif mode == 'check':
        new = build('IJ')
        ref = pd.read_csv('results/nhanes_clean.csv')
        ref['SEQN'] = ref['SEQN'].astype(int)
        print(f"rebuilt {len(new)} rows; reference {len(ref)} rows")
        both = ref.merge(new, on='SEQN', suffixes=('_ref', '_new'))
        print(f"in both: {len(both)}; only reference: {len(set(ref.SEQN) - set(new.SEQN))}; "
              f"only rebuilt: {len(set(new.SEQN) - set(ref.SEQN))}")
        for c in ref.columns[1:]:
            a, b = both[c + '_ref'], both[c + '_new']
            diff = ~np.isclose(a, b)
            print(f"  {c:15s} mismatches {diff.sum():5d}   prevalence ref {a.mean():.3f} new {b.mean():.3f}")
    else:
        out = build('GH')
        out.to_csv('results/nhanes_clean_2011.csv', index=False)
        print(f"wrote results/nhanes_clean_2011.csv {out.shape}")
        print(out.iloc[:, 7:].mean().round(3).to_string())
