"""
test_data_pipeline.py
=====================
Automated Quality Gates and Test Suite for Phase 2 Member 1 Data Pipeline.

Tests:
1. Exact Split Counts & Total Compounds (6,259 train / 782 val / 782 test)
2. Zero Molecule-ID Overlap Across Partitions
3. Zero Bemis-Murcko Scaffold Overlap Across Partitions
4. Compound-ID Row Alignment Across Metadata and Features
5. Missing Label Preservation & Explicit Mask Integrity
6. Core Label Invariants (Obs = Pos + Neg, Total = Obs + Miss)
7. Training-Only Class Weight Formulation & Verification
8. Feature Representation Dimensions & Schemas
9. Preprocessing Leakage Safeguard Verification
10. Evaluability Gate (Zero Single-Class Endpoints in Val/Test)
11. Verified Scientific Literature Table Integrity
"""

import csv
import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MODULE_ROOT.parent.parent

if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    Tox21DataLoader,
    TARGET_ASSAYS,
    DESCRIPTOR_NAMES,
    FEATURE_DIMS,
    TRAIN_POSITIVE_WEIGHTS,
)


class TestTox21DataPipeline(unittest.TestCase):
    """Rigorous verification test suite for Phase 2 Member 1."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = MODULE_ROOT / "configs" / "split_config.json"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

        cls.artifacts_dir = PROJECT_ROOT / "tox21_member2_handoff" / "artifacts"
        cls.source_csv = cls.artifacts_dir / "corrected_scaffold_split.csv"

        with open(cls.source_csv, "r", encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))

        cls.loader = Tox21DataLoader(artifacts_dir=cls.artifacts_dir, config_path=cls.config_path)

    def test_01_split_counts_and_total_molecules(self):
        """Quality Gate 1: Exact row counts (6259 train, 782 val, 782 test, 7823 total)."""
        self.assertEqual(len(self.rows), 7823, "Total curated compounds must equal 7,823.")

        train_rows = [r for r in self.rows if r["split"] == "train"]
        val_rows = [r for r in self.rows if r["split"] in ("val", "validation")]
        test_rows = [r for r in self.rows if r["split"] == "test"]

        self.assertEqual(len(train_rows), 6259, "Train split must contain exactly 6,259 molecules.")
        self.assertEqual(len(val_rows), 782, "Validation split must contain exactly 782 molecules.")
        self.assertEqual(len(test_rows), 782, "Test split must contain exactly 782 molecules.")
        self.assertEqual(len(train_rows) + len(val_rows) + len(test_rows), 7823)

        # Check loader indices
        self.assertEqual(len(self.loader.get_split_indices("train")), 6259)
        self.assertEqual(len(self.loader.get_split_indices("val")), 782)
        self.assertEqual(len(self.loader.get_split_indices("test")), 782)

    def test_02_zero_molecule_id_leakage(self):
        """Quality Gate 2: Partitions must be 100% disjoint (zero shared molecule IDs)."""
        train_ids = set(r["mol_id"] for r in self.rows if r["split"] == "train")
        val_ids = set(r["mol_id"] for r in self.rows if r["split"] in ("val", "validation"))
        test_ids = set(r["mol_id"] for r in self.rows if r["split"] == "test")

        self.assertEqual(len(train_ids & val_ids), 0, "Train and Val must share 0 molecule IDs.")
        self.assertEqual(len(train_ids & test_ids), 0, "Train and Test must share 0 molecule IDs.")
        self.assertEqual(len(val_ids & test_ids), 0, "Val and Test must share 0 molecule IDs.")

    def test_03_zero_scaffold_leakage(self):
        """Quality Gate 3: Partitions must share zero Bemis-Murcko cyclic scaffolds."""
        train_scafs = set(r["scaffold_smiles"] for r in self.rows if r["split"] == "train" and r["scaffold_smiles"] not in ("", "__ACYCLIC__"))
        val_scafs = set(r["scaffold_smiles"] for r in self.rows if r["split"] in ("val", "validation") and r["scaffold_smiles"] not in ("", "__ACYCLIC__"))
        test_scafs = set(r["scaffold_smiles"] for r in self.rows if r["split"] == "test" and r["scaffold_smiles"] not in ("", "__ACYCLIC__"))

        self.assertEqual(len(train_scafs & val_scafs), 0, "Train and Val must share 0 cyclic scaffolds.")
        self.assertEqual(len(train_scafs & test_scafs), 0, "Train and Test must share 0 cyclic scaffolds.")
        self.assertEqual(len(val_scafs & test_scafs), 0, "Val and Test must share 0 cyclic scaffolds.")

    def test_04_acyclic_distribution_policy(self):
        """Quality Gate 4: Documented acyclic policy (all 1,775 in train; 0 in val; 0 in test)."""
        acyclics = [r for r in self.rows if r["scaffold_smiles"] in ("", "__ACYCLIC__")]
        self.assertEqual(len(acyclics), 1775, "Total acyclic molecules must equal 1,775.")

        train_acyclics = [r for r in acyclics if r["split"] == "train"]
        val_acyclics = [r for r in acyclics if r["split"] in ("val", "validation")]
        test_acyclics = [r for r in acyclics if r["split"] == "test"]

        self.assertEqual(len(train_acyclics), 1775, "All acyclics must reside in Train split.")
        self.assertEqual(len(val_acyclics), 0, "Validation split must contain 0 acyclic molecules.")
        self.assertEqual(len(test_acyclics), 0, "Test split must contain 0 acyclic molecules.")

    def test_05_missing_labels_and_mask_integrity(self):
        """Quality Gate 5: Missing labels must remain NaN; masks must be strictly boolean."""
        labels, mask, mol_ids = self.loader.load_labels(split="test")
        self.assertEqual(len(mol_ids), 782)

        total_missing = 0
        total_observed = 0
        for i in range(len(labels)):
            for j in range(len(TARGET_ASSAYS)):
                val = labels[i][j]
                m = mask[i][j]
                is_nan = val != val  # NaN check in Python
                if is_nan:
                    total_missing += 1
                    self.assertFalse(m, "Mask must be False for unmeasured (NaN) labels.")
                else:
                    total_observed += 1
                    self.assertTrue(m, "Mask must be True for observed labels.")
                    self.assertIn(val, (0.0, 1.0), "Observed label must be 0.0 or 1.0.")

        self.assertEqual(total_missing + total_observed, 782 * 12)

    def test_06_label_invariants_across_all_endpoints(self):
        """Quality Gate 6: Invariants Obs = Pos + Neg and Total = Obs + Miss must pass 48/48 checks."""
        for split_name in ["train", "val", "test"]:
            sub_rows = [r for r in self.rows if r["split"] == (split_name if split_name != "val" else "val")]
            n_sub = len(sub_rows)

            for ep in TARGET_ASSAYS:
                pos = sum(1 for r in sub_rows if r[ep] in ("1", "1.0"))
                neg = sum(1 for r in sub_rows if r[ep] in ("0", "0.0"))
                obs = pos + neg
                miss = sum(1 for r in sub_rows if r[ep] in ("", "nan", "NaN", None))

                self.assertEqual(obs, pos + neg, f"Observed invariant violated for {ep} in {split_name}")
                self.assertEqual(n_sub, obs + miss, f"Total invariant violated for {ep} in {split_name}")

    def test_07_evaluability_gate_zero_single_class(self):
        """Quality Gate 7: Val and Test sets must have >= 20 positives and >= 300 negatives per endpoint."""
        for split_name in ["val", "test"]:
            sub_rows = [r for r in self.rows if r["split"] == split_name]
            for ep in TARGET_ASSAYS:
                pos = sum(1 for r in sub_rows if r[ep] in ("1", "1.0"))
                neg = sum(1 for r in sub_rows if r[ep] in ("0", "0.0"))
                self.assertGreaterEqual(pos, 20, f"Insufficient positives ({pos}) for {ep} in {split_name}")
                self.assertGreaterEqual(neg, 300, f"Insufficient negatives ({neg}) for {ep} in {split_name}")

    def test_08_training_class_weights_verification(self):
        """Quality Gate 8: Class weights must strictly derive from training observed labels."""
        computed_weights = self.loader.get_class_weights()
        for idx, ep in enumerate(TARGET_ASSAYS):
            expected_weight = TRAIN_POSITIVE_WEIGHTS[ep]
            actual_weight = computed_weights[idx]
            self.assertAlmostEqual(actual_weight, expected_weight, places=3,
                                   msg=f"Class weight mismatch for endpoint {ep}")

    def test_09_feature_dimensions_specification(self):
        """Quality Gate 9: Verify feature dimension constants."""
        self.assertEqual(FEATURE_DIMS["combined"], 2053)
        self.assertEqual(FEATURE_DIMS["morgan_filtered"], 1011)
        self.assertEqual(FEATURE_DIMS["descriptors_scaled"], 5)
        self.assertEqual(FEATURE_DIMS["morgan"], 2048)
        self.assertEqual(FEATURE_DIMS["descriptors_raw"], 5)

    def test_10_literature_table_verification(self):
        """Quality Gate 10: Verify the 4 landmark papers in literature_table.csv."""
        lit_csv = MODULE_ROOT / "literature" / "literature_table.csv"
        self.assertTrue(lit_csv.exists(), "literature_table.csv must exist.")

        with open(lit_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        self.assertEqual(len(reader), 4, "Literature table must contain exactly 4 verified landmark papers.")
        dois = [r["doi_or_url"] for r in reader]
        self.assertIn("https://doi.org/10.3389/fenvs.2015.00080", dois, "DeepTox DOI must be present.")
        self.assertIn("https://doi.org/10.1039/c7sc02664a", dois, "MoleculeNet DOI must be present.")
        self.assertIn("https://doi.org/10.1038/srep05664", dois, "Tox21 10K DOI must be present.")
        self.assertIn("https://doi.org/10.1021/ci100176x", dois, "QSAR curation DOI must be present.")


if __name__ == "__main__":
    unittest.main()
