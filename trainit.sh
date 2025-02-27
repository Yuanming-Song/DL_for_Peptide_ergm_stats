# Load necessary modules
module load anaconda/2024.06
module load gcc/11.2.0
# Change to the project directory
cd /dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide
source /opt/apps/anaconda/2024.06/etc/profile.d/conda.sh

conda init
# Activate the conda environment located in your writable directory
conda activate dl_py309


# Run the main script
python main_seq_dist.py
