#!/usr/bin/env bash
# Run QC-SCM detection service (development).
# From repo root: ./Boxes/flow/scripts/run_dev.sh
# Or from Boxes/flow: python main.py
# Uses conda env: qc (lowercase)

set -e
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate qc
fi
cd "$(dirname "$0")/.."
exec python main.py
