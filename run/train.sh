#!/bin/bash
#SBATCH --job-name=oral_cancer_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run training
# You can override params like: --model.n_estimators=200
python3 "$PROJECT_ROOT/scripts/train.py" \
  --data.test_size=0.3 \
  --model.n_estimators=100 \
  --model.seed=42 \
  "$@"
