#!/bin/bash

# --- Parse input arguments ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu_name> <python_script> [script_args...]"
    exit 1
fi

GPU_NAME=$1       # e.g., newton1
PYTHON_SCRIPT=$2  # e.g., train.py
shift 2
SCRIPT_ARGS="$@"  # remaining arguments

# --- SLURM Parameters ---
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="PILOT"
MAIL_USER="mohammed-wa@campus.technion.ac.il"
MAIL_TYPE=ALL

# --- Conda parameters ---
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=mpilot-py310

# --- Submit to SLURM ---
sbatch \
    -N $NUM_NODES \
    -c $NUM_CORES \
    -w $GPU_NAME \
    --gres=gpu:$NUM_GPUS \
    --job-name $JOB_NAME \
    --mail-user $MAIL_USER \
    --mail-type $MAIL_TYPE \
    -o "slurm-%N-%j.out" \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run the Python script with parameters
python $PYTHON_SCRIPT $SCRIPT_ARGS

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF
