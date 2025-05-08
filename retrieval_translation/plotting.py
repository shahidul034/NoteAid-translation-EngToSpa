# plotting.py
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns

def plot_metrics(results_dir="results", out_dir="plots", metrics=None):
    """
    Create plots comparing raw vs reranked metrics.
    
    Args:
        results_dir: Directory containing results files
        out_dir: Directory to save plots
        metrics: List of metrics to plot (default: BLEU)
    """
    if metrics is None:
        metrics = ["BLEU"]
        
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(results_dir) if f.startswith("metrics_")]
    
    if not files:
        print(f"No metric files found in {results_dir}")
        return
        
    # Extract configuration details from filenames
    config_data = []
    for fn in files:
        cfg = fn.replace("metrics_", "").replace(".json", "")
        try:
            # If the format is metrics_summary_METHOD_TEMPLATE_SHOTS.json
            parts = cfg.split("_")
            if len(parts) == 4 and parts[0] == "summary": # Check if 'summary' is the first part
                _, method, template, shots_str = parts # Unpack, ignoring the first part
                shots = int(shots_str)
            elif len(parts) == 3: # Original expected format
                 method, template, shots_str = parts
                 shots = int(shots_str)
            else:
                print(f"Skipping file with unexpected name format: {fn}")
                continue

            metric_data = json.load(open(os.path.join(results_dir, fn)))
            
            # Create entry for each metric
            for metric in metrics:
                if metric in metric_data["raw"] and metric in metric_data["rerank"]:
                    config_data.append({
                        "method": method,
                        "template": template,
                        "shots": shots, # Use the converted integer
                        "metric": metric,
                        "raw_score": metric_data["raw"][metric],
                        "rerank_score": metric_data["rerank"][metric],
                        "improvement": metric_data["rerank"][metric] - metric_data["raw"][metric]
                    })
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue
            
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(config_data)
    
    if df.empty:
        print("No valid data found for plotting")
        return
        
    # 1. Plot overall comparison by method
    for metric in metrics:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        methods = metric_df["method"].unique()
        
        for i, method in enumerate(methods):
            method_df = metric_df[metric_df["method"] == method]
            x = np.arange(len(method_df)) + i * 0.3
            plt.bar(x, method_df["raw_score"], width=0.3, label=f"{method} (raw)")
            plt.bar(x + 0.3, method_df["rerank_score"], width=0.3, label=f"{method} (rerank)")
            
        plt.xlabel("Configuration")
        plt.ylabel(f"{metric} Score")
        plt.title(f"{metric} Comparison by Method")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_comparison.png"))
        plt.close()
        
    # 2. Plot improvement heatmap by method and template
    for metric in metrics:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue
            
        # Pivot to create a heatmap
        pivot_df = metric_df.pivot_table(
            index="method", 
            columns="template", 
            values="improvement",
            aggfunc="mean"
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, cmap="RdYlGn", center=0)
        plt.title(f"{metric} Improvement (Rerank - Raw)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_improvement_heatmap.png"))
        plt.close()
        
    # 3. Plot by shots for each method
    for metric in metrics:
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            continue
            
        for method in metric_df["method"].unique():
            method_df = metric_df[metric_df["method"] == method]
            
            plt.figure(figsize=(12, 6))
            for template in method_df["template"].unique():
                template_df = method_df[method_df["template"] == template].sort_values("shots")
                
                plt.plot(template_df["shots"], template_df["raw_score"], 
                         marker='o', linestyle='-', label=f"{template} (raw)")
                plt.plot(template_df["shots"], template_df["rerank_score"], 
                         marker='s', linestyle='--', label=f"{template} (rerank)")
                
            plt.xlabel("Number of shots")
            plt.ylabel(f"{metric} Score")
            plt.title(f"{metric} by Shots for {method}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{metric}_{method}_by_shots.png"))
            plt.close()

if __name__ == "__main__":
    # Example usage
    plot_metrics(
        results_dir="experiment_results/sentence/local_pritamdeka_S_PubMedBert_MS_MARCO/metrics",
        out_dir="experiment_results/sentence/local_pritamdeka_S_PubMedBert_MS_MARCO",
        metrics=["BLEU", "CHRF++", "COMET"]
    )