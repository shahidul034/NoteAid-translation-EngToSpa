# analysis.py
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def compare_runs(path1, path2, metric="BLEU"):
    """
    Compare two runs by calculating differences between their metrics.
    
    Args:
        path1: Path to first results file
        path2: Path to second results file
        metric: Metric to compare (default: "BLEU")
        
    Returns:
        List of tuples (id, score_diff) sorted by absolute difference
    """
    try:
        data1 = json.load(open(path1))
        data2 = json.load(open(path2))
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        return []
        
    diffs = []
    for a, b in zip(data1, data2):
        try:
            score1 = a["metrics"][metric]
            score2 = b["metrics"][metric]
            diffs.append((a["id"], score1 - score2))
        except KeyError:
            # Handle case where metrics aren't found in the expected structure
            continue
            
    diffs_sorted = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)
    # top 10 problematic
    return diffs_sorted[:10]

def aggregate_metrics(results_dir, metrics=None):
    """
    Aggregate metrics from all result files in a directory.
    
    Args:
        results_dir: Directory containing metric files
        metrics: List of metrics to include (default: all metrics)
        
    Returns:
        DataFrame with aggregated metrics
    """
    if metrics is None:
        metrics = ["BLEU", "CHRF++", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F", "COMET"]
        
    files = [f for f in os.listdir(results_dir) if f.startswith("metrics_")]
    rows = []
    
    for fn in files:
        try:
            path = os.path.join(results_dir, fn)
            config = fn.replace("metrics_", "").replace(".json", "")
            method, template, shots = config.split("_")
            
            data = json.load(open(path))
            row = {
                "method": method,
                "template": template,
                "shots": int(shots)
            }
            
            # Add raw metrics
            for m in metrics:
                if m in data["raw"]:
                    row[f"raw_{m}"] = data["raw"][m]
                    
            # Add rerank metrics
            for m in metrics:
                if m in data["rerank"]:
                    row[f"rerank_{m}"] = data["rerank"][m]
                    
            rows.append(row)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue
            
    return pd.DataFrame(rows)

def plot_comparison(results_dir, metric="BLEU", output_dir="plots"):
    """
    Plot comparison of raw vs reranked results.
    
    Args:
        results_dir: Directory containing results
        metric: Metric to plot (default: "BLEU")
        output_dir: Where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get aggregated data
    df = aggregate_metrics(results_dir)
    
    # Group by method and template
    methods = df["method"].unique()
    templates = df["template"].unique()
    
    # Plot by method
    for method in methods:
        method_df = df[df["method"] == method]
        plt.figure(figsize=(12, 6))
        
        for template in templates:
            template_df = method_df[method_df["template"] == template]
            if template_df.empty:
                continue
                
            template_df = template_df.sort_values("shots")
            plt.plot(template_df["shots"], template_df[f"raw_{metric}"], 
                     marker='o', linestyle='-', label=f"{template} (raw)")
            plt.plot(template_df["shots"], template_df[f"rerank_{metric}"], 
                     marker='s', linestyle='--', label=f"{template} (reranked)")
        
        plt.title(f"{method} - {metric} Comparison")
        plt.xlabel("Number of Examples (shots)")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_{metric}_comparison.png"))
        plt.close()

if __name__ == "__main__":
    # Example usage
    plot_comparison("retrieval_translation/results", "BLEU", "retrieval_translation/plots")
    
    # Compare specific runs
    problematic = compare_runs(
        "retrieval_translation/results/raw_dense_T1_5.json",
        "retrieval_translation/results/rerank_dense_T1_5.json"
    )
    print("Top differences between raw and reranked:")
    for idx, diff in problematic:
        print(f"ID {idx}: Difference of {diff:.4f}")