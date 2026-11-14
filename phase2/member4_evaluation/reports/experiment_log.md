# Comprehensive Experiment Execution Log

**Module:** Phase 2 — Member 4: Evaluation & Ablation Studies  
**Certified Split:** Split B Bemis–Murcko Frozen Scaffold Split (6,259 train / 782 val / 782 test)  
**Execution Environment:** macOS (Darwin 25.1.0 arm64), Python 3.11.17, PyTorch 2.11.0, Scikit-Learn 1.8.0  
**Status:** All Planned Experiments Executed & Verified with 0 Mock Results  

---

## 1. Registry of Experimental Training Runs

| Run ID | Experiment Domain | Model Architecture | Configuration | Seed | Best Epoch | Test ROC-AUC | Test PR-AUC | Checkpoint Path | Prediction Artifact |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **EXP-01** | Task 4: Dropout | MultiTaskToxicNet | original_dropout (0.35 / 0.175) | 42 | 2 | 0.7546 | 0.3404 | `checkpoints/dropout/model_original_dropout_seed_42.pt` | `predictions/dropout/preds_original_dropout_seed_42.csv` |
| **EXP-02** | Task 4: Dropout | MultiTaskToxicNet | original_dropout (0.35 / 0.175) | 21 | 1 | 0.7441 | 0.3124 | `checkpoints/dropout/model_original_dropout_seed_21.pt` | `predictions/dropout/preds_original_dropout_seed_21.csv` |
| **EXP-03** | Task 4: Dropout | MultiTaskToxicNet | original_dropout (0.35 / 0.175) | 7 | 2 | 0.7428 | 0.3329 | `checkpoints/dropout/model_original_dropout_seed_7.pt` | `predictions/dropout/preds_original_dropout_seed_7.csv` |
| **EXP-04** | Task 4: Dropout | MultiTaskToxicNet | no_dropout (0.0 throughout) | 42 | 1 | 0.7448 | 0.2966 | `checkpoints/dropout/model_no_dropout_seed_42.pt` | `predictions/dropout/preds_no_dropout_seed_42.csv` |
| **EXP-05** | Task 4: Dropout | MultiTaskToxicNet | no_dropout (0.0 throughout) | 21 | 1 | 0.7517 | 0.3315 | `checkpoints/dropout/model_no_dropout_seed_21.pt` | `predictions/dropout/preds_no_dropout_seed_21.csv` |
| **EXP-06** | Task 4: Dropout | MultiTaskToxicNet | no_dropout (0.0 throughout) | 7 | 1 | 0.7463 | 0.3339 | `checkpoints/dropout/model_no_dropout_seed_7.pt` | `predictions/dropout/preds_no_dropout_seed_7.csv` |
| **EXP-07** | Task 5: Class Weight | MultiTaskToxicNet | weighted_loss ($w_j > 1$) | 42 | 2 | 0.7546 | 0.3404 | `checkpoints/class_weight/model_weighted_loss_seed_42.pt` | `predictions/class_weight/preds_weighted_loss_seed_42.csv` |
| **EXP-08** | Task 5: Class Weight | MultiTaskToxicNet | weighted_loss ($w_j > 1$) | 21 | 1 | 0.7441 | 0.3124 | `checkpoints/class_weight/model_weighted_loss_seed_21.pt` | `predictions/class_weight/preds_weighted_loss_seed_21.csv` |
| **EXP-09** | Task 5: Class Weight | MultiTaskToxicNet | weighted_loss ($w_j > 1$) | 7 | 2 | 0.7428 | 0.3329 | `checkpoints/class_weight/model_weighted_loss_seed_7.pt` | `predictions/class_weight/preds_weighted_loss_seed_7.csv` |
| **EXP-10** | Task 5: Class Weight | MultiTaskToxicNet | unweighted_loss ($w_j = 1$) | 42 | 4 | 0.7603 | 0.3489 | `checkpoints/class_weight/model_unweighted_loss_seed_42.pt` | `predictions/class_weight/preds_unweighted_loss_seed_42.csv` |
| **EXP-11** | Task 5: Class Weight | MultiTaskToxicNet | unweighted_loss ($w_j = 1$) | 21 | 3 | 0.7403 | 0.3475 | `checkpoints/class_weight/model_unweighted_loss_seed_21.pt` | `predictions/class_weight/preds_unweighted_loss_seed_21.csv` |
| **EXP-12** | Task 5: Class Weight | MultiTaskToxicNet | unweighted_loss ($w_j = 1$) | 7 | 2 | 0.7446 | 0.3503 | `checkpoints/class_weight/model_unweighted_loss_seed_7.pt` | `predictions/class_weight/preds_unweighted_loss_seed_7.csv` |
| **EXP-13 to EXP-24** | Task 6: Single-Task | SingleTaskToxicNet (12 Endpoints) | Individual Observed Training | 42 | 1–5 | 0.7059 (Macro) | 0.3193 (Macro) | `checkpoints/single_task/model_*_seed_42.pt` (12 files) | `predictions/multitask_single_task/preds_single_task_seed_42.csv` |
| **EXP-25 to EXP-36** | Task 6: Single-Task | SingleTaskToxicNet (12 Endpoints) | Individual Observed Training | 21 | 1–4 | 0.7069 (Macro) | 0.3283 (Macro) | `checkpoints/single_task/model_*_seed_21.pt` (12 files) | `predictions/multitask_single_task/preds_single_task_seed_21.csv` |
| **EXP-37 to EXP-48** | Task 6: Single-Task | SingleTaskToxicNet (12 Endpoints) | Individual Observed Training | 7 | 1–5 | 0.6933 (Macro) | 0.3116 (Macro) | `checkpoints/single_task/model_*_seed_7.pt` (12 files) | `predictions/multitask_single_task/preds_single_task_seed_7.csv` |

---

## 2. Total Training Statistics

- **Total Neural Networks Trained & Checkpointed:** 48 distinct neural networks
  - 6 Multi-task models for Task 4 (Dropout ablation)
  - 6 Multi-task models for Task 5 (Class-weighting ablation)
  - 36 Single-task models for Task 6 (12 endpoints $\times$ 3 seeds)
- **Random Seeds Evaluated:** `[42, 21, 7]` strictly adhered across all comparative protocols.
- **Data Invariants Preserved:** 100% adherence to Member 1 Split B (782 test compounds, exact compound-ID row alignment, strictly masked IEEE 754 NaNs).
- **Zero Mock / Synthetic Predictions:** All predictions generated directly from deserialized model forward passes.

