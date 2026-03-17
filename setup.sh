#!/bin/bash

echo "----------------------------------------"
echo "ECG Transformer Research Project Setup"
echo "----------------------------------------"

echo "Creating directories"

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/splits

mkdir -p results/figures
mkdir -p results/tables

mkdir -p logs

echo "Creating virtual environment"

python3 -m venv venv

source venv/bin/activate

echo "Upgrading pip"

pip install --upgrade pip

echo "Installing dependencies"

pip install -r requirements.txt

echo "Installing PhysioNet downloader"

pip install wfdb

echo "----------------------------------------"
echo "Downloading ECG datasets"
echo "----------------------------------------"

cd data/raw

echo "Downloading PTB-XL"

python - <<EOF
import wfdb
wfdb.dl_database("ptb-xl", "ptbxl")
EOF

echo "Downloading MIT-BIH Arrhythmia"

python - <<EOF
import wfdb
wfdb.dl_database("mitdb", "mitbih")
EOF

echo "Downloading INCART dataset"

python - <<EOF
import wfdb
wfdb.dl_database("incartdb", "incart")
EOF

cd ../../

echo "----------------------------------------"
echo "Dataset download complete"
echo "----------------------------------------"

echo "Project setup finished"

echo "To activate environment run:"
echo "source venv/bin/activate"

echo "Then start training with:"
echo "python training/train.py"