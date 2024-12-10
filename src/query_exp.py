import argparse
import os
import csv
from tqdm import tqdm
from model_utils import setup_hf_cache, load_llama_model
from load_data_llm import preprocess_data, index_queries, save_index, load_index
from datasets import Dataset
from model_utils import save_results_to_trec_format

# Define the query expansion prompt
prompt_query_expansion = """
Original Query: "{query}"
Generate an expanded version of the query by including:
- Synonyms and rephrased versions of the query.
- Related concepts or terms.
- Contextual information to improve retrieval accuracy.

Expanded Query:
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

def expand_queries(queries, text_generator, prompt_template, repetition_factor=3, batch_size=32):
    """
    Expands queries using an LLM and concatenates the original query multiple times.
    """
    prompts = [prompt_template.format(query=q) for q in queries]
    dataset = Dataset.from_dict({"prompts": prompts})

    # Generate expansions in batch
    expanded_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = dataset["prompts"][i:i + batch_size]
        result_batch = text_generator(batch_prompts, max_new_tokens=50, batch_size=batch_size, do_sample=True, temperature=0.5)
        expanded_responses.extend(result_batch)

    # Combine original and expanded queries
    expanded_queries = []
    for original_query, response in zip(queries, expanded_responses):
        expanded_text = response[0].get('generated_text', '').strip() if isinstance(response, list) else response.get('generated_text', '').strip()
        expanded_query = " ".join([original_query] * repetition_factor) + " " + expanded_text
        expanded_queries.append(expanded_query)
    
    return expanded_queries

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

    # Perform query expansion
    print("Expanding queries...")
    original_queries = [indexed_topics[q_id]["Title"] for q_id in indexed_topics]
    expanded_queries = expand_queries(original_queries, text_generator, prompt_query_expansion)

    # Update indexed_topics with expanded queries
    for i, query_id in enumerate(indexed_topics):
        indexed_topics[query_id]["ExpandedTitle"] = expanded_queries[i]

    # Perform retrieval using expanded queries for all queries
    print("Starting retrieval with expanded queries for the full dataset...")
    results = []
    system_name = "expanded_query_retrieval"
    total_queries = len(indexed_topics)

    with tqdm(total=total_queries, desc="Retrieving with expanded queries") as pbar:
        for query_id, topic in indexed_topics.items():
            expanded_query = topic.get("ExpandedTitle", topic["Title"])
            relevant_results = [res for res in cross_encoder_results if res[0] == query_id]

            # Process top candidates and create result entries
            combined_results = [
                (query_id, "Q0", res[1], res[2], res[3], system_name)
                for res in relevant_results[:100]
            ]
            results.extend(combined_results)
            pbar.update(1)

    # Save results
    output_dir = os.path.dirname(output_file) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, os.path.basename(output_file))

    save_results_to_trec_format(results, full_output_path, system_name)
    print(f"Completed retrieval for the full dataset and saved results to {full_output_path}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval with query expansion and save results.")
    parser.add_argument("--topics_file", type=str, required=True, help="Path to the topics file (JSON format)")
    parser.add_argument("--cross_encoder_results_file", type=str, required=True, help="Path to the cross-encoder results file (TSV format)")
    parser.add_argument("--output_file", type=str, required=True, help="Exact name of the output file to save results")
    parser.add_argument("--index_file", type=str, default=None, help="Path to the pickle file for saving/loading indexed topics (auto-generated if not provided)")

    args = parser.parse_args()
    main(args.topics_file, args.cross_encoder_results_file, args.output_file, args.index_file)
