#!/bin/bash
#SBATCH --job-name=causal_test
#SBATCH --output=causal_test_%j.out
#SBATCH --error=causal_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu_short

# ============================================================================
# CAUSAL DISCOVERY TEST JOB - Ruche HPC (CentraleSupelec)
# ============================================================================

echo "Job started on $(hostname) at $(date)"

# 1. Load the specific Python module available on Ruche
module purge
module load python/3.9.10/gcc-11.2.0

# 2. Activate your virtual environment
# Assumes you created it with: python3 -m venv .venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Could not find .venv or venv directory."
    exit 1
fi

# 3. specific python dependency check (optional, but good for debugging)
python3 --version

# 4. Run the script
# The script will run in the directory where you typed sbatch
echo "Running test_causal_discovery.py..."
python3 test_causal_discovery.py

echo "Job finished at $(date)"
