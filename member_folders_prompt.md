# Reorganise Phase 2 Folders Member-Wise (with meaningful names)

I am working on a college project called **Molecular Toxicity Prediction using the Tox21 dataset**, built by a team of 5 members.

**The problem:** the Phase 2 folders were recently renamed to function names (commit `2a11d75`, "rename member folders to function names"). The names are clean, but now nobody can tell which member built which part — and I need that for our viva and for the contributions section.

**What I want:** folder names that show **both the member and what the part does**, plus one file that maps every folder and file to its owner.

---

## STEP 0 — Do This First

1. Confirm `git status` is clean. If anything is uncommitted, commit it before starting.
2. Create a new branch: `git checkout -b member-wise-structure`.
3. Read `phase2/experiments/PART_A_RESULTS.md` and the root `README.md` so you understand what each part does before renaming it.

---

## STEP 1 — Attribution (I already worked this out from git history)

I ran `git log --diff-filter=A` to find who originally added each folder. Use this table. **Do not guess attribution from folder names or file contents.**

| Current folder / file | Original author (git) | Member name | What it does |
|---|---|---|---|
| `phase2/data/` | `Aggarwalmansi` | Mansi Aggarwal | Tox21 data pipeline: loader, splits, masks |
| `phase2/deep_learning/` | `nachiket7-dev` | Nachiketa | Multi-task deep learning model, training, embeddings |
| `phase2/hybrid/` | `tanisha800` | Tanisha Dhiman | ML + DL hybrid model (RF/ExtraTrees on learned features) |
| `phase2/evaluation/` | `aditisingh60` | Aditi Singh | Evaluation, ablation studies, error analysis |
| `phase2/experiments/` | `tanisha800` | Tanisha Dhiman | Part A improvements, checkpoint fix, ensembles |
| `train_models.py` (Phase 1) | `adamyatiwari12` | Adamya Tiwari | Phase 1 baseline classifiers, 12 endpoints |
| `evaluate.py`, `results/`, `app.py` (Phase 1) | `nachiket7-dev` | Nachiketa | Phase 1 evaluation, statistics, Streamlit dashboard |
| `tox21_project/` (Phase 1) | `tanisha800` | Tanisha Dhiman | Dataset, EDA, preprocessing, scaffold split |
| `tox21_member2_handoff/` (Phase 1) | `Aggarwalmansi` | Mansi Aggarwal | Feature engineering and handoff artifacts |

**Two things to check with me before you rely on this table:**

1. **A git author is whoever committed the code, which is not always who wrote it.** Show me this table and let me confirm or correct it before you rename anything.
2. **Adamya Tiwari has no Phase 2 folder.** Git shows only Phase 1 baseline work and a README edit. So a strict "one folder per member" structure would leave an empty folder. Do not invent one. Ask me what Adamya's Phase 2 contribution was, and if there is none, say so honestly in the contributions file.

---

## STEP 2 — Rename the Phase 2 Folders

Use `git mv` so history is preserved. Pattern: `memberN_firstname_function`.

```
phase2/data/          ->  phase2/member1_mansi_data_pipeline/
phase2/deep_learning/ ->  phase2/member2_nachiketa_deep_learning/
phase2/hybrid/        ->  phase2/member3_tanisha_hybrid_model/
phase2/evaluation/    ->  phase2/member4_aditi_evaluation/
phase2/experiments/   ->  phase2/member5_tanisha_improvements/
```

Adjust the numbers and names once I confirm Step 1. If I tell you Adamya owns part of Phase 2, renumber accordingly.

Leave `phase2/results/` as it is — it holds the combined output of everybody's work and belongs to no single member.

---

## STEP 3 — Do Not Reorganise Phase 1

Phase 1 stays exactly where it is. Do not move, rename or modify:

```
train_models.py    evaluate.py    app.py
model_evaluation_results.csv
results/           tox21_project/    tox21_member2_handoff/
requirements.txt   requirements-eval.txt
```

Reason: `app.py` is my live presentation dashboard and loads these by hardcoded path, and Phase 1 is the frozen baseline I compare against. **Phase 1 authorship goes in the contributions file (Step 5), not into folder names.**

---

## STEP 4 — Fix Everything the Rename Breaks

This is the risky part. The previous rename had to update about 27 path references, and this one will too.

For each folder you rename:
1. `grep -rn "<old_path>" --include=*.py --include=*.md --include=*.json .` across the whole repo (skip `.venv` and `.git`).
2. Update every import, string path and documentation reference you find.
3. Check `docs/*.md` too — those files describe the old folder names.

Then verify, and do not report success until all of these pass:
- every script under `phase2/` runs
- every test under `phase2/*/tests/` passes
- saved models and checkpoints still load
- `streamlit run app.py` runs and every tab loads
- `python -m phase2.<new_path>.<script>` style module imports still resolve

---

## STEP 5 — Create `CONTRIBUTIONS.md` (this is the part that actually solves my problem)

Folder names only show ownership at the top level. I want a single file I can open to see who made what, down to the file level.

Create one `CONTRIBUTIONS.md` at the repo root with:

1. **A member table** — name, GitHub username, phase, role, and the folders/files they own.
2. **A file-level table** for anything non-obvious — the main scripts in each folder and who wrote them.
3. **A short paragraph per member** in simple English saying what they built and what it contributes to the final result.
4. **A note on the numbering clash:** Phase 1 and Phase 2 use different member numbers. In `app.py` Member 3 means the baseline classifiers (Adamya), while in `phase2/` Member 3 means the hybrid model (Tanisha). Say this plainly so no examiner is confused.
5. **How the attribution was determined** — one line saying it came from git history, verified with the team.

Keep it factual. If ownership of something is genuinely shared or unknown, write "shared" or "unclear" rather than picking a name.

---

## STEP 6 — Add an Author Line to Each Main File

For the main script in each member folder, add or update the author in the existing module docstring at the top. For example:

```python
"""
hybrid_model.py
Member 3 — Tanisha Dhiman
Trains one tree ensemble per toxicity endpoint on the learned DL features.
"""
```

Only the main files (the ones a viva examiner would open) — do not touch every utility file, and do not add a banner comment to 38 files.

---

## STEP 7 — Update the README

In the root `README.md`:
- Update the folder-structure section to the new names.
- Replace the team contributions section with a short summary table and a link to `CONTRIBUTIONS.md`.
- Do not duplicate the full contributions content in both files.

---

## RULES

- **Do not delete anything.** Use `git mv` to move, and `archive/` for anything unwanted. Ask me before any real delete — my environment blocks deletes by default, and a blocked delete can leave `.git` lock files that break later git commands.
- **Do not change any model, result, metric or experiment.** This task is only about names, structure and documentation. No number in any results file should change.
- Keep the code simple and readable. Do not refactor code while renaming folders — renaming and refactoring in one commit makes it impossible to review.
- Commit in small steps: one commit for the renames, one for the import fixes, one for the documentation. Not one giant commit.
- Do not create extra folders, duplicate READMEs or new documentation beyond `CONTRIBUTIONS.md`.

---

## WORK IN THIS ORDER

1. Show me the attribution table from Step 1 and your proposed folder names.
2. **Wait for my approval.**
3. Rename with `git mv`, then fix every broken reference.
4. Run all the checks in Step 4 and show me the output.
5. Write `CONTRIBUTIONS.md`, add the author docstrings, update the README.
6. Confirm `git status` is clean and no results file changed.
