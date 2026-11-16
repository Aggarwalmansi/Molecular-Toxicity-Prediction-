# Setup, inference and reproduction — T028, Section D

The current model architecture, settings and result table are documented in the [root README](../../README.md). All current deliverables use the hybrid modeling module’s current experiment summary; earlier-network ablations are labelled historical.

## Install and run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m phase2.member5_report_ppt.prepare_models
streamlit run app.py
```

The preparation command restores the residual checkpoint, exports stronger raw-feature forests using training labels, reuses the evaluated feature-fusion forests, records matching embeddings and hashes, and evaluates validation/test predictions. It does not select settings on test. Forest settings and alpha remain fixed from the validation study.

## Predict

Open **Predict a molecule**, choose a preset or enter valid SMILES, then select **Predict all 12 endpoints**. Four methods produce 48 scores: Stronger RF, Multitask DL, Feature fusion, Probability blend. The blend is 0.80 RF + 0.20 DL. Download includes scores, class, threshold and canonical SMILES. The default threshold is 0.5 and scores are not calibrated human toxicity probabilities.

```bash
python -m phase2.member5_report_ppt.predict --smiles 'CCO' --output predictions.csv
python -m unittest discover -s phase2/member5_report_ppt/tests -v
```

## Files

- `predict.py`: frozen saved-model inference; no training during user requests.
- `prepare_models.py`: reproducible serving bundle preparation and endpoint metrics.
- `models/bundle.json`: checkpoint SHA-256, alpha, forest configuration and versions.
- `models/inference_metrics.csv`: measured serving-bundle validation/test metrics.
- `models/checkpoint_test_embeddings.parquet`: embedding parity fixture.
- `report/main.tex`: project report with a separate team cover page and IEEE two-column body.
- `presentation/Molecular_Toxicity_T028_refined.pptx`: current 12-slide presentation.

Do not edit result CSVs to force agreement. Local arithmetic or package differences must be investigated or explicitly labelled; published experiment tables and local measured metrics retain their own sources.

## Individual contribution areas

Data contract/literature, neural modeling, fusion/blending, experimental evaluation, and inference/report/presentation integration are the five work areas. Team members: Adamya Tiwari (2401010025), Tanisha Dhiman (2401010475), Aditi Singh (2401020006), Nachiket Amlekar (2401020039), Mansi Agarwal (2401010264). Contributions: Mansi Agarwal — data pipeline; Nachiket Amlekar — deep learning; Tanisha Dhiman — hybrid modeling; Aditi Singh — evaluation; Adamya Tiwari — application, report and presentation.

Reporting convention: endpoint metrics are rounded to four decimals before macro averaging, matching the current evaluation scripts. Full-precision averaging may differ in the last displayed decimal.
