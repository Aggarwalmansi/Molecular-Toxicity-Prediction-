# Tox21 Preprocessing Project

This project prepares the Tox21 toxicity dataset for machine learning.
It inspects the data, checks its quality, canonicalizes the SMILES and makes a reproducible scaffold split.
It does **not** compute fingerprints or descriptors and does **not** train models.

## Data source

* File: `tox21 3.csv`

## Files

| File | Content |
|---|---|
| `eda.ipynb` | Main notebook: every step explained with outputs and charts |
| `data_pipeline.py` | Loading, column/label checks, missing-label statistics, RDKit canonicalization, duplicate & conflict checks, summary table |
| `scaffold_split.py` | Bemis–Murcko scaffolds, scaffold grouping, seeded split, leakage checks, class distribution per split |
| `dataset_summary.csv` | All statistics in long format: `section, assay, metric, value` |
| `split_manifest.csv` | One row per original CSV row: IDs, original & canonical SMILES, scaffold, scaffold group ID, split or exclusion |
| `train.csv` / `validation.csv` / `test.csv` | `mol_id`, original SMILES, canonical SMILES, 12 labels (missing labels left empty) |
| `exclusion_report.csv` | The invalid SMILES with RDKit's error message (excluded from all splits) |
| `duplicate_audit.csv` | Audit trail of duplicate structures / stereoisomer groups and any conflicting labels |
| `requirements.txt` | Minimal Python packages |

## How to run in Google Colab

1. Open <https://colab.research.google.com> and choose **File → Upload notebook** → `eda.ipynb`.
2. Run the first code cell. It installs RDKit if it is missing.
3. Run the next cell. An **upload** button appears. Upload `tox21 3.csv`, `data_pipeline.py` and `scaffold_split.py` together.
   (Alternatively, clone the GitHub repository and `%cd` into this folder.)
4. Choose **Runtime → Run all**. The output files are written to `/content/`. Download them from the Files panel on the left.

## How to run locally

```bash
pip install -r requirements.txt
jupyter notebook eda.ipynb
```

## Key results (from the executed notebook)

* 7,831 rows. All 12 assay columns contain only 0, 1 or missing.
* 77,946 observed and 16,026 missing labels (17.05% of 93,972 label cells). Missing labels are **never** replaced by 0.
* Strong class imbalance: 2.9–16.2% positives per assay (negative-to-positive ratio 5–34).
* No exact duplicate rows, repeated IDs, repeated SMILES strings or repeated canonical structures.
* 8 invalid SMILES (all with hypervalent `[AlH3]`). They are excluded and listed in `exclusion_report.csv`. 7,823 valid molecules remain.
* 107 groups (229 molecules) are stereoisomers of each other. 31 of these groups have different observed labels in at least one assay. They are flagged, not changed.
* Scaffold split, seed 42: train 6,258 (79.99%), validation 782 (10.00%), test 783 (10.01%). All leakage checks pass.

## Duplicate and conflicting-label policy

1. The original CSV is never changed, and nothing is silently deleted (the manifest has every row).
2. Structurally equivalent molecules (same canonical SMILES) and stereoisomers are always placed in the **same** split.
3. Conflicting observed labels are **flagged** in `duplicate_audit.csv` for investigation. No label is picked arbitrarily
   and no majority vote is used. Stereoisomers can genuinely differ in activity.
4. Invalid SMILES are excluded from splitting and documented in `exclusion_report.csv`.

## Scaffold split policy

* Scaffolds: RDKit Bemis–Murcko scaffold, computed **without stereochemistry**.
* **Empty scaffolds:** 1,775 valid molecules (22.7%) have no ring, so their scaffold is `""`. Putting them in one giant group
  would force all of them into training, leaving none for validation and test. Instead, each distinct acyclic structure
  (ignoring stereochemistry) forms its own group. These groups are distributed by the seeded allocation below.
  *Consequence:* for acyclic molecules the split behaves like a structure-grouped random split, not a "novel scaffold" split.
  The `has_ring_scaffold` column in `split_manifest.csv` lets you evaluate the two groups separately.
* **Allocation (seed 42):** groups larger than half the validation target are "big" (only the benzene scaffold here).
  Big and small groups are each shuffled with `random.Random(42)`. Then each group, big ones first, goes to train if it
  still fits under 80%, otherwise to validation if it fits under 10%, otherwise to test.
* Proportions are close to, but not exactly, 80/10/10, because whole groups are moved together.
* Positive rates differ between splits (e.g. NR-AR test 8.7% vs train 3.6%). This is a normal side effect of scaffold splitting.
  Some validation/test assays have only ~13–40 positives, so their evaluation scores will be noisy.
