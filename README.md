# MinGuard: artifacts

Code, workloads, guard scores and human labels for the paper
*MinGuard: Calibrated Pre-Execution Data-Minimization Guards for LLM Database Agents* (ICDE 2027 submission).

## Contents

| Path | What it is |
|---|---|
| `src/dbmeta.py` | Personal-data annotation of the seven BIRD databases (168 of 620 columns, seven categories) and the role policies |
| `src/sqlfacts.py` | Deterministic fact layer: lineage, aggregation, small-aggregate count, bounded dry run |
| `src/guards.py` | Short-circuit rules, the single-fact questions, max fusion, and all LLM/judge baselines |
| `src/evaluate.py` | Question-grouped conformal risk control and question-clustered bootstrap |
| `src/make_tables.py`, `src/make_figures.py`, `src/paper_data.py` | Regenerate every table and data figure of the paper from the score files |
| `data/` | The five workloads (`pilot_bench`, `heldout_bench`, `real_bench*`, `final_bench`) and the natural output distribution |
| `runs/` | Scores of every guard on every request, plus the inputs of the robustness experiments |
| `annotation/` | Human labels, each request labeled by two annotators, with the blank and filled sheets: 200 design-database requests, a random sample of 150 new-database requests, and all 211 judged requests of the New DBs test half (`human_labels_newdb_test.json` holds the 165 not in the earlier sample). A third annotator labeled the 74 New DBs requests without agreement, blind to the first two; `human_labels_newdb_adjudicated.json` holds the majority labels used in the paper (rule fixed in `src/make_annotation_adjudication.py` before the labels came back) |

Comments in the code are partly in Chinese.

## Reproducing the tables

1. Download the BIRD dev set (`dev_20240627`) from https://bird-bench.github.io/ and unpack it to `data/dev_20240627/`, so that
   `data/dev_20240627/dev_databases/<db>/<db>.sqlite` and `data/dev_20240627/dev.json` exist.
2. `pip install -r requirements.txt`
3. `bash reproduce.sh` writes all tables to `tables/` and the three data figures (Figs. 3--5) to `figures/`
   (about 20 minutes on 8 cores; the bootstraps and the 200-split risk control dominate). No API key and no GPU are needed: all model outputs are in `runs/`.

## Re-running the guards

`src/run_guards.py` calls the hosted Jev 1.13 (`jev-1.13-20260917`) and the LLM judges through OpenRouter
(`OPENROUTER_API_KEY`). `src/jev27b_jobs.py`, `src/jev27b_run.py` and `src/jev27b_score.py` run the open JEV-27B
(https://huggingface.co/autotrust/JEV-27B) locally with vLLM 0.31 on one 80 GB GPU. `src/readout_guard.py` and
`src/train_clf.py` produce the Qwen3.5-4B readout and the ModernBERT classifiers. `src/pg_import.py` and
`src/pg_bench.py` measure the fact layer on PostgreSQL 16. `src/scope_combo.py` substitutes DeepSeek-V3.2's answer to the `scope`
question into MinGuard offline (Sec. VII-H, `runs/scope_combo.json`).

## Labels

`annotation/human_labels.json` and `annotation/human_labels_newdb.json` hold one record per request with the
annotators' verdicts, violation types, whether the candidate answers the question, and a free-text note. The paper counts
four types as over-disclosure (added personal attribute, added identifier, statistic answered by a listing, extra
people) and excludes the ambiguous types (filter or sort value, wrong people, other). All labels were assigned by people.

## Fonts

`figures/fonts/` holds Nunito and Fredoka (SIL Open Font License 1.1), used by the figures.

## License

Code: MIT. Annotations and scores: CC BY 4.0. The BIRD databases are not redistributed here; see their own license.
