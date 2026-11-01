#!/usr/bin/env bash
# Downloads the public dataset files used in the paper into ./data
#   UNSW-NB15 official training/testing partition (Moustafa & Slay, 2015)
#   CICIDS2017 stratified sample published by Yang et al. (MTH-IDS, IEEE IoT-J 2022)
# The original sources are https://research.unsw.edu.au/projects/unsw-nb15-dataset
# and https://www.unb.ca/cic/datasets/ids-2017.html; the GitHub mirrors below are
# used because they are scriptable.
set -euo pipefail
cd "$(dirname "$0")/../data"
mkdir -p unsw cic
U=https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification
curl -fL -o unsw/UNSW_NB15_training-set.csv "$U/UNSW_NB15_training-set.csv"
curl -fL -o unsw/UNSW_NB15_testing-set.csv  "$U/UNSW_NB15_testing-set.csv"
curl -fL -o cic/CICIDS2017_sample.csv \
  https://raw.githubusercontent.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/main/data/CICIDS2017_sample.csv
sha256sum unsw/*.csv cic/*.csv
