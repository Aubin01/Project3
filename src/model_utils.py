import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

def setup_hf_cache():
    """Sets up the Hugging Face cache directory to use the instructor's shared path."""
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/behrooz.mansouri/HF'
    print("Hugging Face cache directory set to:", os.environ['TRANSFORMERS_CACHE'])

def load_llama_model():
    """
    Loads the LLaMA-3.1-8B-Instruct model and returns the model pipeline.
    Ensures that padding and device configurations are set correctly.
    """
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Load tokenizer and set padding configurations
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set model loading configurations
    try:
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        print(f"Loaded model {model_id} on {'GPU' if device == 0 else 'CPU'}.")
    except Exception as e:
        print("Error loading model:", e)
        return None

    # Define text generation pipeline with specific generation configurations
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.pad_token_id
    )
    
    return text_generator
        
def generate_response(prompt, text_generator, max_tokens=200):
    """
    Generates a response for a given prompt using the loaded model.
    """
    try:
        print(f"Generating response for prompt: {prompt[:50]}...")
        result = text_generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.9, top_p=0.8)
        response_text = result[0]['generated_text']
        response_split = response_text.split("Assistant:")
        response = response_split[1].strip() if len(response_split) > 1 else response_text.strip()
        print("Response generated successfully.")
        return response
    except Exception as e:
        print("Error generating response:", e)
        return None

def expand_queries_with_llm(queries, text_generator, prompt_template):
    """
    Expands queries using a large language model and appends the expanded terms to each query.
    """
    expanded_queries = {}
    for query_id, query_text in queries.items():
        prompt = prompt_template.format(query=query_text)
        expanded_query = generate_response(prompt, text_generator, max_tokens=50)
        expanded_queries[query_id] = f"{query_text} {expanded_query}"
    print(f"Expanded {len(queries)} queries.")
    return expanded_queries

def pairwise_rerank(llm_model, query, doc_pairs):
    """
    Uses LLM to rerank document pairs based on their relevance to the query.
    """
    reranked_docs = []
    for doc1, doc2 in doc_pairs:
        prompt = f"""
        Query: {query}
        Document 1: {doc1}
        Document 2: {doc2}
        Which document is more relevant to the query? Answer with '1' or '2'.
        """
        response = generate_response(prompt, llm_model, max_tokens=5)
        winner = doc1 if response.strip() == "1" else doc2
        reranked_docs.append(winner)
    print(f"Reranked {len(doc_pairs)} document pairs.")
    return reranked_docs

def save_results_to_trec_format(results, output_file, run_name):
    """
    Saves the results in TREC format for evaluation.
    """
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results to the specified file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        results.sort(key=lambda x: (x[0], x[3]))  # Sort by query_id and rank
        query_count = {}
        for query_id, _, answer_id, rank, score, system_name in results:
            if query_count.get(query_id, 0) < 100:
                writer.writerow([query_id, "Q0", answer_id, rank, score, system_name])
                query_count[query_id] = query_count.get(query_id, 0) + 1
                
    print(f"Results saved to {output_file}.")

