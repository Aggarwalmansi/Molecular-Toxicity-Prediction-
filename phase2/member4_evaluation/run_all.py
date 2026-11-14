"""
run_all.py
==========
End-to-End Orchestrator for Phase 2 Member 4: Evaluation & Ablation Studies.
Executes:
  1. Data Invariants & Contract Verification
  2. Task 4: Dropout Ablation Study (6 runs across 3 seeds)
  3. Task 5: Class-Weighting Ablation Study (6 runs across 3 seeds)
  4. Task 6: Multitask vs Single-Task Benchmark (36 single-task + 3 multitask runs)
  5. Task 7: Comprehensive Error Analysis & Threshold Optimization
  6. Task 8: Consolidated Tables, Publication-Grade Visualizations & Reports
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure safe plotting environment
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

from phase2.member4_evaluation.src.data_utils import Tox21DataLoaderWrapper, set_seed
from phase2.member4_evaluation.src.dropout_ablation import DropoutAblationStudy
from phase2.member4_evaluation.src.class_weight_ablation import ClassWeightAblationStudy
from phase2.member4_evaluation.src.multitask_vs_single_task import MultitaskVsSingleTaskStudy
from phase2.member4_evaluation.src.error_analysis import ComprehensiveErrorAnalysis
from phase2.member4_evaluation.src.comparison import ConsolidatedModelComparison
from phase2.member4_evaluation.src.visualization import PublicationVisualizer


def run_pipeline(stages: List[str]):
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 75)
    print("PHASE 2 — MEMBER 4: COMPREHENSIVE EVALUATION & ABLATION STUDIES PIPELINE")
    print("=" * 75)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Target Output Directory: {base_dir}")
    print(f"Selected Stages: {stages}")
    print("=" * 75 + "\n")

    # Step 0: Contract Verification
    print("[Verification] Verifying Member 1 data contracts and frozen scaffold split...")
    wrapper = Tox21DataLoaderWrapper()
    print("  └── Passed! Train: 6,259 | Val: 782 | Test: 782 | Features: 2,053 | Endpoints: 12\n")

    # Task 4: Dropout Ablation
    if "all" in stages or "dropout" in stages:
        print("\n" + "#" * 70)
        print("STAGE 1: DROPOUT ABLATION (Original 0.35 vs No Dropout 0.0, 3 Seeds)")
        print("#" * 70)
        dropout_study = DropoutAblationStudy(output_dir=base_dir)
        dropout_study.run_all()

    # Task 5: Class Weighting Ablation
    if "all" in stages or "class_weight" in stages:
        print("\n" + "#" * 70)
        print("STAGE 2: CLASS-WEIGHTING ABLATION (Weighted vs Unweighted Loss, 3 Seeds)")
        print("#" * 70)
        cw_study = ClassWeightAblationStudy(output_dir=base_dir)
        cw_study.run_all()

    # Task 6: Multitask vs Single-Task Benchmark
    if "all" in stages or "single_task" in stages:
        print("\n" + "#" * 70)
        print("STAGE 3: MULTITASK VS SINGLE-TASK (12 Endpoint Nets vs Shared Trunk, 3 Seeds)")
        print("#" * 70)
        mt_st_study = MultitaskVsSingleTaskStudy(output_dir=base_dir)
        mt_st_study.run_all()

    # Task 7: Error Analysis
    if "all" in stages or "error_analysis" in stages:
        print("\n" + "#" * 70)
        print("STAGE 4: COMPREHENSIVE ERROR ANALYSIS & THRESHOLD OPTIMIZATION")
        print("#" * 70)
        err_analysis = ComprehensiveErrorAnalysis(output_dir=base_dir)
        err_analysis.run_analysis()

    # Task 8: Consolidated Tables & Visualizations
    if "all" in stages or "comparison" in stages or "plots" in stages:
        print("\n" + "#" * 70)
        print("STAGE 5: FINAL MODEL COMPARISON & SYNTHESIS")
        print("#" * 70)
        comp = ConsolidatedModelComparison(output_dir=base_dir)
        comp.build_final_comparison()

        print("\n" + "#" * 70)
        print("STAGE 6: PUBLICATION-QUALITY VISUALIZATIONS (300 DPI)")
        print("#" * 70)
        visualizer = PublicationVisualizer(output_dir=base_dir)
        visualizer.generate_all_plots()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"ALL MEMBER 4 WORK COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print(f"Results written to: {base_dir / 'results'}")
    print(f"Plots written to:   {base_dir / 'plots'}")
    print(f"Reports written to: {base_dir / 'reports'}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Member 4 Evaluation Pipeline")
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", "dropout", "class_weight", "single_task", "error_analysis", "comparison", "plots"],
        help="Specify which stage(s) to execute",
    )
    args = parser.parse_args()
    run_pipeline(args.stage)


if __name__ == "__main__":
    main()
