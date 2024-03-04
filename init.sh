echo "=> Installing conda"
operating_system="$(uname)" 
architecture="$(uname -m)" 
miniforge_source="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-${operating_system}-${architecture}.sh" 
curl -L ${miniforge_source} -o install_miniforge.sh && \
bash install_miniforge.sh -b -p conda
rm install_miniforge.sh
conda/bin/conda init
source ~/.bashrc

echo "=> Updating conda's package solver to use mamba"
conda config --set solver libmamba

echo "=> Setting up CUQUANTUM_ROOT"
conda env config vars set CUQUANTUM_ROOT=${CONDA_PREFIX}

echo "=> Installing cuQuantum"
conda install -c conda-forge -y cuquantum

echo "=> Installing cuQuantum-python"
conda install -c conda-forge -y cuquantum-python
