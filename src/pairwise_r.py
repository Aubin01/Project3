import argparse
import os
import csv
from tqdm import tqdm
from model_utils import setup_hf_cache, load_llama_model
from load_data_llm import preprocess_data, index_queries, save_index, load_index
from datasets import Dataset
from model_utils import save_results_to_trec_format

# Define zero-shot prompt
prompt_zero_shot = """
You are a helpful travel assistant. Answer the user's question accurately and concisely.

User: "{question}"
Assistant:
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
    from datasets import Dataset

    # Create all pairwise combinations
    pairs = [
        (doc1_id, doc1_text, doc2_id, doc2_text)
        for (doc1_id, doc1_text), (doc2_id, doc2_text) in combinations(candidates, 2)
    ]

    # Prepare prompts for the model
    prompts = [
        f"""
        Query: {query}
        Document A: {doc1_text}
        Document B: {doc2_text}
        Which document is more relevant to the query? Answer with 'A' or 'B'.
        """
        for _, doc1_text, _, doc2_text in pairs
    ]

    # Create a dataset for batch processing
    dataset = Dataset.from_dict({"prompts": prompts})

    # Perform batch inference
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

def main(topics_file, cross_encoder_results_file, output_file, index_file=None):
    setup_hf_cache()

    # Dynamically name the index file based on the topics file name
    topics_file_base = os.path.basename(topics_file).split(".")[0]
    index_file = index_file or f"indexed_{topics_file_base}.pkl"
    
    # Load or create index
    if os.path.exists(index_file):
        print(f"Loading existing index from {index_file}")
        indexed_topics = load_index(index_file)
    else:
        print(f"Creating new index for {topics_file}")
        topics, _, _ = preprocess_data(topics_file)
        indexed_topics = index_queries(topics)
        save_index(indexed_topics, index_file)
        print(f"Saved new index to {index_file}")
    
    # Load Cross-Encoder results
    cross_encoder_results = load_results(cross_encoder_results_file)

    # Load the text generation model
    text_generator = load_llama_model()

    # Perform pairwise re-ranking for all queries
    print("Starting retrieval with pairwise re-ranking...")
    results = []
    total_queries = len(set(res[0] for res in cross_encoder_results))
    
    with tqdm(total=total_queries, desc="Re-ranking queries") as pbar:
        current_query_id = None
        batch_candidates = []

        for query_id, answer_id, rank, score in cross_encoder_results:
            question = indexed_topics.get(query_id, {}).get("Title", "")
            if not question:
                continue

            if query_id != current_query_id and current_query_id is not None:
                if batch_candidates:
                    # Perform pairwise re-ranking
                    ranked_candidates = pairwise_re_rank(question, batch_candidates, text_generator)

                    # Add ranked results to the final output
                    results.extend([
                        (current_query_id, "Q0", doc_id, rank + 1, 1.0 / (rank + 1), "pairwise_ranking")
                        for rank, (doc_id, _) in enumerate(ranked_candidates[:100])
                    ])

                batch_candidates.clear()
                pbar.update(1)

            current_query_id = query_id
            batch_candidates.append((answer_id, indexed_topics[query_id]["Title"]))

        # Handle the last batch
        if batch_candidates:
            ranked_candidates = pairwise_re_rank(question, batch_candidates, text_generator)
            results.extend([
                (current_query_id, "Q0", doc_id, rank + 1, 1.0 / (rank + 1), "pairwise_ranking")
                for rank, (doc_id, _) in enumerate(ranked_candidates[:100])
            ])
            pbar.update(1)

    # Dynamically handle output file path
    output_dir = os.path.dirname(output_file) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, os.path.basename(output_file))

    save_results_to_trec_format(results, full_output_path, "pairwise_ranking")
    print(f"Completed retrieval and saved results to {full_output_path}.\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval with zero-shot prompt and save results.")
    parser.add_argument("--topics_file", type=str, required=True, help="Path to the topics file (JSON format)")
    parser.add_argument("--cross_encoder_results_file", type=str, required=True, help="Path to the cross-encoder results file (TSV format)")
    parser.add_argument("--output_file", type=str, required=True, help="Exact name of the output file to save results")
    parser.add_argument("--index_file", type=str, default=None, help="Path to the pickle file for saving/loading indexed topics (auto-generated if not provided)")

    args = parser.parse_args()
    main(args.topics_file, args.cross_encoder_results_file, args.output_file, args.index_file)

