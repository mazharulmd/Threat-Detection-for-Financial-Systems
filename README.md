# HALX — Privacy-Preserving Hybrid Attention-LSTM + XGBoost Threat Detection for Financial Networks

Code, experiments and IEEE conference paper for **HALX**, a network-flow threat detector for
financial infrastructure that combines:

1. **A privacy-preserving inference layer** — direct-identifier suppression, Mondrian
   multidimensional *k*-anonymity on quasi-identifiers, and (ε, δ)-differentially-private
   standardization statistics (Gaussian mechanism, zCDP accounting).
2. **An Attention-LSTM encoder** — each flow record is read as a sequence of feature tokens by a
   2-layer BiLSTM, followed by multi-head self-attention and attention pooling (the pooling
   weights double as per-alert explanations).
3. **An XGBoost scoring head** — scores `[embedding ‖ privatized features]`, outputs P(attack),
   the attack family, and a low/medium/high severity tier.

```
paper/            IEEE conference paper (LaTeX, IEEEtran) — main.tex, sections/, figures/, generated/
halx/             library: data.py (loading/splitting/SMOTE), privacy.py, models.py, train.py
experiments/      run.py (main / ablation / privacy / latency), make_report.py (tables + plot data)
scripts/          download_data.sh, run_all.sh
results/raw/      one JSON per (experiment, dataset, config, seed)
```

## Paper

`paper/main.pdf` — 6-page IEEE conference paper (IEEEtran). All numbers in the paper are generated
from `results/raw/*.json` by `experiments/make_report.py` (macros in `paper/generated/numbers.tex`).

## Headline results (mean ± std over 3 seeds, test sets, all methods on identical privatized inputs)

| Dataset | Method | F1 (%) | FPR (%) | AUC (%) |
|---|---|---|---|---|
| CICIDS2017 subset | XGBoost | 99.68 ± 0.06 | 0.56 ± 0.13 | 99.96 |
| | Transformer | 99.15 ± 0.23 | 1.78 ± 0.51 | 99.86 |
| | CNN-LSTM | 98.04 ± 0.16 | 3.86 ± 0.40 | 99.57 |
| | **HALX** | 99.58 ± 0.07 | 0.77 ± 0.11 | 99.95 |
| UNSW-NB15 (official split) | XGBoost | 89.13 ± 0.10 | 27.18 ± 0.33 | 98.26 |
| | Transformer | 89.23 ± 1.78 | 26.18 ± 7.31 | 98.09 |
| | CNN-LSTM | 88.66 ± 0.26 | 28.98 ± 0.82 | 98.17 |
| | **HALX** | 89.11 ± 0.07 | 27.21 ± 0.14 | 98.25 |

Honest summary: HALX matches a tuned XGBoost in accuracy (it does not beat it), clearly outperforms
the pure deep models on CICIDS2017, has the most stable false-positive rate on UNSW-NB15, the highest
detection rate at a 1% false-alarm budget on UNSW-NB15 (86.9% vs 85.9% for XGBoost, seed 0), 97% precision in its
high-severity tier, and adds per-alert attention explanations. The privacy layer (k = 10, ε = 1)
costs ≤ 0.16 F1 points. The high UNSW-NB15 FPR at τ = 0.5 is caused by the known train/test shift of
the official split. See the paper for ablations, the privacy–utility sweep and latency.

## Reproduce

```bash
pip install -r requirements.txt
./scripts/download_data.sh            # UNSW-NB15 official split + CICIDS2017 stratified subset
./scripts/run_all.sh                  # all experiments (CPU only; several hours on 4 cores)
python -m experiments.make_report     # regenerates paper/generated/*
cd paper && latexmk -pdf main.tex
```

Individual experiments:

```bash
python -m experiments.run main     --dataset unsw --seeds 0 1 2
python -m experiments.run ablation --dataset cic  --seeds 0 1 2
python -m experiments.run privacy  --dataset cic  --seeds 0 --model HALX
python -m experiments.run latency  --dataset unsw
```

Results are cached per run in `results/raw/`, so interrupted runs resume where they stopped.

## Datasets

* **UNSW-NB15** (Moustafa & Slay, 2015) — official training/testing partition
  (175,341 / 82,332 flows). The official test set is used only for final reporting.
* **CICIDS2017** (Sharafaldin et al., 2018) — the 56,661-flow stratified subset released by
  Yang et al. (MTH-IDS, IEEE IoT-J 2022), split 70/15/15.

Both are public research datasets; see the original sources for their licences.

## Notes on methodology

* Flows are modelled independently (feature tokens), **not** as sliding windows of consecutive
  flows: the public UNSW-NB15 partition is sorted by label and has no timestamps, so windowing
  would leak labels through row order.
* The DP guarantee covers the released normalization statistics only; network weights are not
  trained with DP-SGD. See Sec. VII of the paper.
