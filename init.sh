#!/bin/bash

echo "=> Installing conda"
operating_system="$(uname)" 
architecture="$(uname -m)" 
miniforge_source="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-${operating_system}-${architecture}.sh" 
curl -L ${miniforge_source} -o install_miniforge.sh && \
bash install_miniforge.sh -b -p conda
rm install_miniforge.sh
conda/bin/conda init
source /root/.bashrc

echo "=> Updating conda's package solver to use mamba"
conda/bin/conda config --set solver libmamba

echo "=> Installing cuQuantum"
conda/bin/conda install cuda -c nvidia
conda/bin/conda install -c conda-forge -y cuquantum

echo "=> Setting up CUQUANTUM_ROOT"
conda/bin/conda env config vars set CUQUANTUM_ROOT=/conda

echo "=> Installing cuQuantum-python"
conda/bin/conda install -c conda-forge -y cuquantum-python 

echo "=> Cloning repositories"
git clone https://github.com/simplysudhanshu/bits_to_qubits.git
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git

echo "=> Installing additional python packages (btq requirements)"
conda/bin/conda install -c conda-forge -y --file bits_to_qubits/requirements.txt 
