#!/usr/bin/env python3
"""
Causal Discovery Test Script for Ruche HPC
This is a minimal test version to verify the code works correctly.
Uses T=50 for quick validation before running full benchmarks.
"""

import json
import time
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Disable display for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
# Use small T for testing - change this for production runs
T_VALUE = int(os.environ.get("T_VALUE", 50))  # Default to 50 for quick test

BASE_DIR = Path(".")
out_dir = BASE_DIR / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "dataset_name": "wt_walks_v1",
    "experiment_name": "actuators_random_walk_1",
    "data_root": str(BASE_DIR / "data"),

    "variables": [
        "load_in",
        "load_out",
        "hatch",
        "rpm_in",
        "rpm_out",
        "current_in",
        "current_out",
        "pressure_downwind",
    ],

    "T": T_VALUE,
    "tau_max": 2,
    "pc_alpha": 0.05,

    "methods": ["pcmci", "pcmciplus"],
    "ci_tests": [
        {"name": "ParCorr", "kwargs": {"significance": "analytic"}},
        {"name": "CMIknn", "kwargs": {"significance": "shuffle_test", "knn": 5}},
    ],

    "fdr_method": "none",
    "score_contemporaneous": True,
    "random_seed": 42,
    "out_dir": str(out_dir),
}

print(f"=" * 80)
print(f"CAUSAL DISCOVERY TEST - T={T_VALUE}")
print(f"=" * 80)

config_path = out_dir / "benchmark_config.json"
config_path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
print(f"Config saved to: {config_path.resolve()}")

# ============================================================================
# STEP 1: LOAD DATA + BUILD TIGRAMITE DATAFRAME
# ============================================================================
print("\n[STEP 1] Loading data...")

try:
    import causalchamber.datasets as cc_datasets
    from tigramite import data_processing as pp
except ImportError as e:
    print(f"ERROR: Missing required packages. Install with:")
    print(f"  pip install causalchamber tigramite")
    sys.exit(1)

data_root = Path(CONFIG["data_root"])
data_root.mkdir(parents=True, exist_ok=True)

print(f"Downloading/loading dataset: {CONFIG['dataset_name']}")
dataset = cc_datasets.Dataset(
    name=CONFIG["dataset_name"],
    root=str(data_root),
    download=True
)

print(f"Available experiments (first 10): {dataset.available_experiments()[:10]}")

exp = dataset.get_experiment(name=CONFIG["experiment_name"])
df = exp.as_pandas_dataframe()

variables = CONFIG["variables"]
T = int(CONFIG["T"])

missing_cols = [c for c in variables if c not in df.columns]
if missing_cols:
    raise ValueError(f"These variables are missing in the dataset: {missing_cols}")

df_sub = df.loc[:, variables].iloc[:T].copy()

if df_sub.isna().any().any():
    df_sub = df_sub.ffill().bfill()

data = df_sub.values.astype(float)
tigramite_df = pp.DataFrame(data=data, var_names=variables)

print(f"Data shape: {tigramite_df.values[0].shape}")
print(f"Variables: {variables}")
print(f"First 5 rows:\n{df_sub.head()}")

# ============================================================================
# STEP 2: RUN BENCHMARK (4 combinations)
# ============================================================================
print("\n[STEP 2] Running causal discovery algorithms...")

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn

tau_max = int(CONFIG["tau_max"])
pc_alpha = float(CONFIG["pc_alpha"])

def run_one(method_name, ci_test_obj):
    pcmci = PCMCI(dataframe=tigramite_df, cond_ind_test=ci_test_obj, verbosity=0)
    t0 = time.time()

    if method_name == "pcmci":
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    elif method_name == "pcmciplus":
        results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
    else:
        raise ValueError("Unknown method: " + method_name)

    runtime = time.time() - t0
    return results, runtime

RUNS = [
    ("pcmci", "ParCorr", ParCorr(significance="analytic")),
    ("pcmciplus", "ParCorr", ParCorr(significance="analytic")),
    ("pcmci", "CMIknn", CMIknn(significance="shuffle_test", knn=5)),
    ("pcmciplus", "CMIknn", CMIknn(significance="shuffle_test", knn=5)),
]

all_results = {}

for method_name, test_name, ci_test in RUNS:
    key = f"{method_name}__{test_name}"
    print(f"\n{'='*60}")
    print(f"Running: {key} | tau_max={tau_max} | pc_alpha={pc_alpha}")

    results, runtime = run_one(method_name, ci_test)
    all_results[key] = {"results": results, "runtime_sec": runtime}

    graph = results["graph"]
    n_links = int((graph != "").sum())

    print(f"Done: {key}")
    print(f"Runtime: {runtime:.2f} sec")
    print(f"Graph shape: {graph.shape}")
    print(f"Discovered links: {n_links}")

print(f"\n{'='*60}")
print(f"All runs finished. Keys: {list(all_results.keys())}")

# ============================================================================
# STEP 3: PRINT SIGNIFICANT LINKS
# ============================================================================
print("\n[STEP 3] Printing significant links...")

alpha_level = float(CONFIG["pc_alpha"])

def make_pcmci_for_test(test_name: str):
    if test_name == "ParCorr":
        ci_test = ParCorr(significance="analytic")
    elif test_name == "CMIknn":
        ci_test = CMIknn(significance="shuffle_test", knn=5)
    else:
        raise ValueError(f"Unknown test: {test_name}")
    return PCMCI(dataframe=tigramite_df, cond_ind_test=ci_test, verbosity=0)

def print_run(key: str):
    results = all_results[key]["results"]
    runtime = all_results[key]["runtime_sec"]

    method_name, test_name = key.split("__", 1)
    pcmci_obj = make_pcmci_for_test(test_name)

    print(f"\n{'='*80}")
    print(f"RUN: {key}")
    print(f"Runtime: {runtime:.2f} sec | alpha_level: {alpha_level} | tau_max: {tau_max}")
    print(f"Discovered links: {(results['graph'] != '').sum()}")

    pcmci_obj.print_significant_links(
        p_matrix=results["p_matrix"],
        val_matrix=results["val_matrix"],
        alpha_level=alpha_level,
    )

for key in ["pcmci__ParCorr", "pcmciplus__ParCorr", "pcmci__CMIknn", "pcmciplus__CMIknn"]:
    print_run(key)

# ============================================================================
# STEP 4: EXPORT EDGELISTS TO CSV
# ============================================================================
print("\n[STEP 4] Exporting edge lists to CSV...")

var_names = list(CONFIG["variables"])

def results_to_edgelist(run_key: str, results: dict, runtime_sec: float, alpha: float):
    graph = results["graph"]
    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]

    rows = []
    n = len(var_names)

    for tgt in range(n):
        for src in range(n):
            for lag_idx in range(tau_max + 1):
                link_type = graph[tgt, src, lag_idx]
                if link_type == "":
                    continue

                rows.append({
                    "run_key": run_key,
                    "method": run_key.split("__", 1)[0],
                    "ci_test": run_key.split("__", 1)[1],
                    "runtime_sec": runtime_sec,
                    "source": var_names[src],
                    "target": var_names[tgt],
                    "lag": lag_idx,
                    "lag_str": f"t-{lag_idx}",
                    "graph_code": link_type,
                    "strength": float(val_matrix[tgt, src, lag_idx]),
                    "p_value": float(p_matrix[tgt, src, lag_idx]),
                    "is_significant": bool(p_matrix[tgt, src, lag_idx] <= alpha),
                })

    edges_all = pd.DataFrame(rows).sort_values(
        ["run_key", "target", "source", "lag"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    edges_sig = edges_all[edges_all["is_significant"]].copy().reset_index(drop=True)
    return edges_all, edges_sig

all_csv_paths = []

for run_key in all_results.keys():
    results = all_results[run_key]["results"]
    runtime = all_results[run_key]["runtime_sec"]

    edges_all, edges_sig = results_to_edgelist(run_key, results, runtime, alpha_level)

    path_all = out_dir / f"edges_all__{run_key}__T{T}.csv"
    path_sig = out_dir / f"edges_sig__{run_key}__T{T}.csv"

    edges_all.to_csv(path_all, index=False)
    edges_sig.to_csv(path_sig, index=False)

    all_csv_paths.append((run_key, str(path_all), str(path_sig), len(edges_all), len(edges_sig)))

summary_df = pd.DataFrame(all_csv_paths, columns=["run_key", "csv_all", "csv_sig", "n_edges_all", "n_edges_sig"])
print(summary_df)
print(f"\nSaved CSVs to: {out_dir.resolve()}")

# ============================================================================
# STEP 5: GROUND TRUTH + METRICS
# ============================================================================
print("\n[STEP 5] Computing metrics against ground truth...")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

truth_graph = np.full((len(var_names), len(var_names), tau_max + 1), "", dtype="<U3")

def add_true_link(source, target, lag, linktype="-->"):
    s = var_names.index(source)
    t = var_names.index(target)
    if lag < 0 or lag > tau_max:
        raise ValueError(f"lag={lag} out of range [0..{tau_max}]")
    truth_graph[t, s, lag] = linktype

# Autocorrelation (lag 1 and 2 for all vars)
for v in var_names:
    for lag in range(1, tau_max + 1):
        add_true_link(v, v, lag, "-->")

# Physics-inspired ground truth links
add_true_link("load_in", "rpm_in", 1)
add_true_link("load_out", "rpm_out", 1)
add_true_link("load_in", "current_in", 1)
add_true_link("load_out", "current_out", 1)
add_true_link("current_in", "rpm_in", 1)
add_true_link("current_out", "rpm_out", 1)
add_true_link("rpm_in", "pressure_downwind", 1)
add_true_link("rpm_out", "pressure_downwind", 1)
add_true_link("hatch", "pressure_downwind", 1)

print(f"Ground truth edges: {int((truth_graph != '').sum())}")

def compute_metrics(pred_graph, true_graph):
    y_true = (true_graph != "").flatten()
    y_pred = (pred_graph != "").flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    shd_presence = int(np.sum((true_graph != "") != (pred_graph != "")))

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd_presence": shd_presence,
    }

rows = []
for run_key in ["pcmci__ParCorr", "pcmciplus__ParCorr", "pcmci__CMIknn", "pcmciplus__CMIknn"]:
    pred_graph = all_results[run_key]["results"]["graph"]
    runtime = all_results[run_key]["runtime_sec"]

    m = compute_metrics(pred_graph, truth_graph)
    m["run_key"] = run_key
    m["runtime_sec"] = float(runtime)
    m["T"] = T
    rows.append(m)

metrics_df = pd.DataFrame(rows).set_index("run_key").sort_index()
print(metrics_df)

out_path = out_dir / f"metrics_T{T}.csv"
metrics_df.to_csv(out_path)
print(f"Saved metrics to: {out_path.resolve()}")

# ============================================================================
# STEP 6: SAVE GRAPHS (without display)
# ============================================================================
print("\n[STEP 6] Saving causal graphs...")

import tigramite.plotting as tp

def threshold_results(results, alpha_level):
    graph = results["graph"].copy()
    val_matrix = results["val_matrix"].copy()
    p_matrix = results["p_matrix"]
    nonsig = (p_matrix > alpha_level)
    graph[nonsig] = ""
    val_matrix[nonsig] = 0.0
    return graph, val_matrix

for run_key in ["pcmci__ParCorr", "pcmciplus__ParCorr", "pcmci__CMIknn", "pcmciplus__CMIknn"]:
    res = all_results[run_key]["results"]
    runtime = all_results[run_key]["runtime_sec"]

    graph_sig, val_sig = threshold_results(res, alpha_level)

    plt.figure(figsize=(9, 7))
    tp.plot_graph(
        val_matrix=val_sig,
        graph=graph_sig,
        var_names=var_names,
        link_colorbar_label="Link strength"
    )
    plt.title(f"{run_key} | T={T} | alpha={alpha_level} | runtime={runtime:.1f}s", fontsize=12)

    fname = out_dir / f"causal_graph__{run_key}__T{T}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"T = {T}")
print(f"Output directory: {out_dir.resolve()}")
print(f"Total runtime summary:")
for run_key in all_results:
    print(f"  {run_key}: {all_results[run_key]['runtime_sec']:.2f} sec")
