import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ranx import Run, Qrels, evaluate, fuse

def load_qrels(qrel_file):
    """Load QREL ground truth from a TREC-formatted file."""
    print(f"Loading QRELs from: {qrel_file}")
    try:
        qrels = Qrels.from_file(qrel_file, kind="trec")
        print(f"Loaded {len(qrels)} QREL entries.")
        return qrels
    except Exception as e:
        print(f"Error loading QREL file: {e}")
        raise

def evaluate_model(qrels, run_file, output_csv, model_name):
    """Evaluate a retrieval model, save metrics to CSV, and plot the ski-jump curve."""
    print(f"Loading run file: {run_file}")
    try:
        run = Run.from_file(run_file, kind="trec")
    except Exception as e:
        print(f"Error loading run file: {e}")
        raise

    print(f"Evaluating {model_name}...")

    # Define evaluation metrics
    metrics = ["precision@1", "precision@5", "precision@10", "ndcg@5", "mrr", "map"]

    try:
        results = evaluate(qrels, run, metrics=metrics, make_comparable=True)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

    # Save metrics to CSV
    df_metrics = pd.DataFrame([results])
    df_metrics.insert(0, 'Model', model_name)
    df_metrics.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")

    print(f"\n=== Evaluation Results for {model_name} ===")
    print(df_metrics)

    # Generate ski-jump plot
    plot_ski_jump(qrels, run, model_name)


def plot_ski_jump(qrels, run, model_name):
    """Plot a ski-jump curve for the precision@5 metric."""
    # Evaluate precision@5 for each query
    results_per_query = evaluate(qrels, run, metrics=["precision@5"], return_mean=False)
    query_ids = list(qrels.qrels.keys())
    per_query_p_at_5 = results_per_query

    # Sort data for a ski-jump plot
    sorted_data = sorted(zip(query_ids, per_query_p_at_5), key=lambda x: x[1])
    sorted_query_ids, sorted_p_at_5 = zip(*sorted_data)

    # Create a DataFrame for structured plotting
    df_p_at_5 = pd.DataFrame({'Query_ID': sorted_query_ids, 'P@5': sorted_p_at_5})

    # Add jitter to P@5 values for visual separation
    jitter = np.random.uniform(-0.01, 0.01, size=len(sorted_p_at_5))
    jittered_p_at_5 = np.array(sorted_p_at_5) + jitter

    # Generate the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_p_at_5['Query_ID'], jittered_p_at_5, 'o', markersize=5)
    plt.title(f'Ski-Jump Plot for P@5 ({model_name})', fontsize=14)
    plt.xlabel('Query IDs', fontsize=12)
    plt.ylabel('P@5', fontsize=12)

    # Limit the number of x-ticks to improve readability
    step = max(1, len(df_p_at_5['Query_ID']) // 20)  # Show one label every 20 queries (adjust as needed)
    plt.xticks(ticks=range(0, len(df_p_at_5['Query_ID']), step), 
               labels=df_p_at_5['Query_ID'][::step], rotation=45, fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot to a file
    plot_file = f"{model_name.lower().replace(' ', '_')}_ski_jump_plot.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Ski-jump plot saved as {plot_file}")
    plt.show()

def fuse_results(run_files, output_file, fusion_method="mnz"):
    """Fuse results from multiple run files."""
    print("Starting result fusion...")
    runs = [Run.from_file(run_file, kind="trec") for run_file in run_files]

    fused = fuse(runs, norm="min-max", method=fusion_method)
    fused.save(output_file, kind="trec")
    print(f"Fused results saved to {output_file}.")

def main(qrel_file, run_files, model_names, output_prefix):
    """Main script to evaluate individual runs and fused results."""
    # Load QREL file
    qrels = load_qrels(qrel_file)

    # Evaluate individual runs
    for run_file, model_name in zip(run_files, model_names):
        output_csv = f"{output_prefix}_{model_name.lower().replace(' ', '_')}_eval.csv"
        evaluate_model(qrels, run_file, output_csv, model_name)

    # Fuse pairwise rerank and query expansion results
    pairwise_file = run_files[0]  # Pairwise rerank file
    query_exp_file = run_files[1]  # Query expansion file
    fused_output_file = f"{output_prefix}_fused_pairwise_queryexp_results.trec"
    fuse_results([pairwise_file, query_exp_file], fused_output_file)

    # Evaluate fused results
    evaluate_model(qrels, fused_output_file, f"{output_prefix}_fused_eval.csv", "Fused Pairwise + Query Expansion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval results and fuse runs.")
    parser.add_argument("--qrel_file", type=str, required=True, help="Path to the QREL file (TREC format).")
    parser.add_argument("--run_files", type=str, nargs=3, required=True, help="Paths to the result files: pairwise, query expansion, pairwise on expanded queries.")
    parser.add_argument("--model_names", type=str, nargs=3, required=True, help="Names of the models for the respective run files.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files.")

    args = parser.parse_args()
    main(args.qrel_file, args.run_files, args.model_names, args.output_prefix)
