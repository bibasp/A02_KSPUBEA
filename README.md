# hydro_tl_ews

**Transfer Learning for Hydrological Early Warning in Data-Scarce Regions.**

An Entity-Aware LSTM (EA-LSTM) is pre-trained on a large donor pool of CAMELS-US
catchments, then transferred to a data-scarce target basin (2-year warmup) and
evaluated under a rolling-origin *walk-forward* protocol. The system produces both
continuous streamflow predictions and flood/drought early-warning probabilities,
with SHAP-based interpretability and a 7-basin multi-regime generalization study.

## Repository layout

```
hydro_tl_ews/
├── src/hydro_tl_ews/            # core library (importable package)
│   ├── data/                    # CAMELS loading, sequence datasets, preprocessing,
│   │                            #   donor selection (clustering.py), synthetic data
│   ├── models/                  # EA-LSTM cell/model (ealstm.py) + losses
│   ├── training/                # trainer, transfer (fine-tune A/B), walk_forward
│   ├── evaluation/              # metrics (NSE/KGE/PBIAS/AUC/Brier/BSS), extreme thresholds
│   ├── xai/                     # SHAP attribution
│   └── utils/                   # config, device, logging, seed
│
├── scripts/                     # runnable entry points
│   ├── run_experiment.py        # single-stage CLI  (--config configs/<stage>.yaml)
│   ├── run_multi_target.py      # 7-basin study orchestrator (resumable)
│   ├── smoke_pipeline.py        # end-to-end synthetic smoke test
│   ├── run_full_pipeline.sh     # runs all stages in order on CAMELS
│   ├── gen_multi_target_configs.py   # generates configs/multi_target/*  (shared: TARGETS)
│   ├── paper_supplement_analysis.py  # shared analysis fns + single-basin supplement
│   ├── stages/                  # per-stage implementations (pretrain, finetune, …)
│   ├── analysis/                # post-processing (drought recal., multi-target supplement, baselines)
│   ├── figures/                 # figure generation (make_figures, make_camels_map)
│   └── paper/                   # PDF / manuscript-DOCX / learning-guide builders
│
├── configs/                     # YAML experiment configs (+ multi_target/ = 7×6)
├── tests/                       # pytest suite
├── docs/                        # manuscript (.docx), paper (.pdf), guides
├── results/                     # all metrics, parquet, checkpoints  (generated)
├── data/                        # CAMELS-US dataset  (not versioned — see data/readme.txt)
└── archive/                     # finished one-off operational scripts (kept for provenance)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# CAMELS-US must be extracted under data/ (see data/readme.txt)
```

## Running

Single stage (config-driven):

```bash
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/pretrain.yaml
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/finetune_conservative.yaml
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/walk_forward.yaml
```

Stages: `pretrain`, `finetune_conservative`, `finetune_progressive`, `local_baseline`,
`zero_shot`, `walk_forward`, `min_data_sensitivity`.

Full CAMELS pipeline / multi-basin study:

```bash
bash scripts/run_full_pipeline.sh                              # all stages, one target
.venv/Scripts/python.exe scripts/run_multi_target.py           # 7-basin study (needs pretrain.pt)
```

Smoke test (synthetic, no data required):

```bash
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/smoke_test.yaml --smoke
.venv/Scripts/python.exe -m pytest -q
```

Rebuild figures and manuscript from `results/`:

```bash
.venv/Scripts/python.exe scripts/figures/make_figures.py
.venv/Scripts/python.exe scripts/paper/build_paper.py
.venv/Scripts/python.exe scripts/paper/build_manuscript_docx.py
```

## Method summary

- **Model** — EA-LSTM (Kratzert et al., 2019): static attributes drive a time-invariant
  input gate; dynamic forcings drive the forget/candidate/output gates. Hidden 256,
  dropout 0.4, forget-bias 3.0.
- **Transfer** — regional pre-train → target fine-tune. *Approach A*: head-only (LSTM
  frozen). *Approach B*: progressive unfreeze of the last 25% of LSTM params at a
  100× smaller LR.
- **Evaluation** — data-scarce walk-forward: 2-year warmup, 90-day refit cadence,
  online causal bias correction, validation tail held out at each refit with
  best-weight restoration.
- **Early warning** — at-site climatological flood/drought thresholds; skill scored by
  AUC, Brier, and Brier Skill Score against a day-of-year climatology benchmark.

## Documentation

- `PROJECT_MASTER_REPORT.txt` — full project reference (objective, results, logs).
- `PROJECT_STATUS_AND_CONTINUATION.txt` — session-by-session status / continuation point.
- `docs/` — manuscript, built PDF, corrections guide, learning guide, review response.
