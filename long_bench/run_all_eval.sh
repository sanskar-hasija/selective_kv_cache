#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate sanskar

export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

# Run the Python script with the provided arguments
python eval.py "$@"