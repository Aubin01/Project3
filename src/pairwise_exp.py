import argparse
import os
import csv
from tqdm import tqdm
from model_utils import setup_hf_cache, load_llama_model
from datasets import Dataset
from model_utils import save_results_to_trec_format

# Define the pairwise re-ranking prompt
prompt_pairwise_ranking = """
Query: {query}
Document A: {doc1}
Document B: {doc2}
Which document is more relevant to the query? Answer with 'A' or 'B'.
"""

def load_results(results_file):
    """Loads results from a TREC-formatted TSV file."""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for query_id, _, answer_id, rank, score, _ in reader:
            results.append((str(query_id), answer_id, int(rank), float(score)))
    print(f"Loaded {len(results)} results from {results_file}")
    return results

def pairwise_re_rank(query, candidates, text_generator, batch_size=32):
    """
    Perform pairwise re-ranking of candidates based on relevance to the query using batch processing.
    """
    from itertools import combinations

    # Create all pairwise combinations
    pairs = [
        (doc1_id, doc1_text, doc2_id, doc2_text)
        for (doc1_id, doc1_text), (doc2_id, doc2_text) in combinations(candidates, 2)
    ]

    # Prepare prompts for the model
    prompts = [
        prompt_pairwise_ranking.format(query=query, doc1=doc1_text, doc2=doc2_text)
        for _, doc1_text, _, doc2_text in pairs
    ]

    # Perform batch inference
    dataset = Dataset.from_dict({"prompts": prompts})
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = dataset["prompts"][i:i + batch_size]
        result_batch = text_generator(batch_prompts, max_new_tokens=5, batch_size=batch_size, do_sample=True, temperature=0.7, top_p=0.9)
        results.extend(result_batch)

    # Collect scores from results
    scores = {doc_id: 0 for doc_id, _ in candidates}
    for result, (doc1_id, _, doc2_id, _) in zip(results, pairs):
        response = result[0].get("generated_text", "").strip().upper() if isinstance(result, list) else result.get("generated_text", "").strip().upper()
        if response == "A":
            scores[doc1_id] += 1
        elif response == "B":
            scores[doc2_id] += 1

    # Sort candidates by their scores
    sorted_candidates = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_candidates

def main(topics_file, expanded_results_file, output_file):
    setup_hf_cache()

    # Load results from the expanded queries retrieval
    expanded_results = load_results(expanded_results_file)

    # Load the text generation model
    text_generator = load_llama_model()

    # Perform pairwise re-ranking
    print("Starting pairwise re-ranking on expanded query results...")
    results = []
    total_queries = len(set(res[0] for res in expanded_results))

    with tqdm(total=total_queries, desc="Pairwise re-ranking queries") as pbar:
        current_query_id = None
        batch_candidates = []

        for query_id, answer_id, rank, score in expanded_results:
            # Use the expanded query as the basis for re-ranking
            expanded_query = f"Expanded Query for Query ID {query_id}"

            if query_id != current_query_id and current_query_id is not None:
                if batch_candidates:
                    # Perform pairwise re-ranking
                    ranked_candidates = pairwise_re_rank(expanded_query, batch_candidates, text_generator)

                    # Add ranked results to the final output
                    results.extend([
                        (current_query_id, "Q0", doc_id, rank + 1, 1.0 / (rank + 1), "pairwise_ranking_expa")
                        for rank, (doc_id, _) in enumerate(ranked_candidates[:100])
                    ])

                batch_candidates.clear()
                pbar.update(1)

            current_query_id = query_id
            batch_candidates.append((answer_id, f"Document text for ID {answer_id}"))

        # Handle the last batch
        if batch_candidates:
            ranked_candidates = pairwise_re_rank(expanded_query, batch_candidates, text_generator)
            results.extend([
                (current_query_id, "Q0", doc_id, rank + 1, 1.0 / (rank + 1), "pairwise_ranking_expa")
                for rank, (doc_id, _) in enumerate(ranked_candidates[:100])
            ])
            pbar.update(1)

    # Save the pairwise re-ranked results
    output_dir = os.path.dirname(output_file) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, os.path.basename(output_file))

    save_results_to_trec_format(results, full_output_path, "pairwise_ranking_expa")
    print(f"Completed pairwise re-ranking and saved results to {full_output_path}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform pairwise re-ranking on expanded query results.")
    parser.add_argument("--topics_file", type=str, required=True, help="Path to the topics file (JSON format).")
    parser.add_argument("--expanded_results_file", type=str, required=True, help="Path to the expanded query results file (TREC format).")
    parser.add_argument("--output_file", type=str, required=True, help="Exact name of the output file to save pairwise re-ranked results.")

    args = parser.parse_args()
    main(args.topics_file, args.expanded_results_file, args.output_file)
