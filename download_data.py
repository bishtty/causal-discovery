#!/usr/bin/env python3
"""
Data Download Script for Ruche HPC

This script should be run on the LOGIN NODE (`ruche01`, `ruche02`).
It connects to the internet to download the required datasets from CausalChamber
and saves them to the local `data/` directory.

The main `test_causal_discovery.py` script can then be run on a compute node
in offline mode, reading from this directory.
"""
import sys
from pathlib import Path
import causalchamber.datasets as cc_datasets

# Configuration must match test_causal_discovery.py
DATASET_NAME = "wt_walks_v1"
DATA_ROOT = Path("./data")

print("=" * 60)
print("Starting data download...")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Target directory: {DATA_ROOT.resolve()}")
print("=" * 60)

# Ensure the target directory exists
DATA_ROOT.mkdir(parents=True, exist_ok=True)

try:
    # This command will download the data if it's not already present
    dataset = cc_datasets.Dataset(
        name=DATASET_NAME,
        root=str(DATA_ROOT),
        download=True
    )
    print(f"\nSuccessfully downloaded/verified dataset '{DATASET_NAME}'.")
    print(f"Data is stored in: {DATA_ROOT.resolve()}")
    print("\nYou can now run the main script on a compute node.")

except Exception as e:
    print(f"\nERROR: An error occurred during download.", file=sys.stderr)
    print(f"  Please check your internet connection and run this script", file=sys.stderr)
    print(f"  on a login node (e.g., ruche01).", file=sys.stderr)
    print(f"\nDETAILS:", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(1)

print("=" * 60)
