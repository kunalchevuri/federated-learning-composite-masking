# Federated Learning and Composite Metric Masking in Wellness Systems

**Paper:** When the Score Looks Fine: Composite Metric Masking 
of Per-Dimension Federated Learning Failures in Wellness Applications  
**Conference:** IEEE BIBM 2026 (under review)  
**Authors:** Kunal Chevuri, Dr. Sergei Chuprov (UTRGV)

## What This Paper Is About

Wellness platforms like Whoop, Garmin, and Fitbit combine 
multiple health scores (sleep, activity, mental health, diet, 
etc.) into one single composite number. This composite can look 
perfectly fine even when one individual dimension is critically 
failing, because strong dimensions pull the average up and hide 
the weak one. We call this the Masking Effect.

This paper formally defines the Masking Gap metric to measure 
how severe the hiding is, proves it is statistically significant 
across 20 demographically diverse federated learning clients, 
and shows that no aggregation method can fix it because the root 
cause is the composite training objective itself.

## Key Contributions

1. Masking Gap metric: composite AUC minus worst per-dimension 
   AUC, with severity tiers (Mild / Moderate / Severe)
2. First empirical proof masking is method-agnostic across 5 FL 
   methods: FedAvg, SCAFFOLD, CS-Agg, Ditto, q-FedAvg
3. Architectural diagnosis: root cause is the composite training 
   objective, not the aggregation mechanism
4. Dimension-weighted loss proof-of-concept validates the correct 
   fix direction

## Dataset

NHANES 2015-2018 (CDC National Health and Nutrition Examination 
Survey). 8,430 usable adult records. 20 federated clients 
partitioned by income quintile x age bracket.

Not included due to CDC data terms of use.  
Download: https://archive.org/download/continuous-nhanes/ContinuousNHANES.zip

## The 7 Wellness Dimensions

| Dimension | NHANES Variable | Poor Threshold |
|---|---|---|
| Sleep Quality | SLD012 | Below 7 hrs/night |
| Physical Activity | PAQ MET computation | Below 600 MET-min/week |
| Sedentary Behavior | PAD680 | Above 360 min/day |
| Mental Health | PHQ-9 (DPQ010-090) | Score 10 or above |
| Metabolic Health | BMXBMI | BMI 30 or above |
| Functional Wellbeing | HSD010 | Fair or Poor response |
| Diet and Nutrition | DR1TSUGR | Above median sugar intake |

## Experiment Conditions

| ID | Method | Description |
|---|---|---|
| C1 | Centralized equal-weight | Ground truth baseline |
| C2 | Centralized weighted loss | Fix direction validation |
| C3 | FedAvg | Standard FL baseline |
| C4 | SCAFFOLD | Drift-corrected FL |
| C5 | CS-Agg | Trust-weighted FL |
| C6 | Ditto | Personalized FL |
| C7 | q-FedAvg | Fairness-aware FL |

## Project Structure

- src/ — all experiment scripts
- results/ — experiment output CSVs (copied from Colab)
- figures/ — all 6 paper figures
- paper/ — methodology section and paper drafts
- docs/ — requirements and experiment plan

## Status

Under review. Repository made public upon acceptance.
