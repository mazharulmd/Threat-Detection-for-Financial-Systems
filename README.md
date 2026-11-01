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
