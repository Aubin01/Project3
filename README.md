# Large Language Models for Information Retrieval (LLM-IR)

`Project Overview`
This project explores the application of Large Language Models (LLMs) for enhancing Information Retrieval (IR) effectiveness. We focus on two primary techniques:

`Query Expansion`: Using LLMs to augment search queries with semantically related terms to improve recall.
`Pairwise Re-ranking`: Utilizing LLMs to re-rank document pairs based on relevance to the query.

The system integrates multiple modules for data preprocessing, query expansion, pairwise re-ranking, and results evaluation using standard IR metrics.

# Project Structure
`Files and Their Functions`

# load_data_llm.py

- Handles loading and preprocessing of topics, Qrel, and other input data.
  Supports query cleaning and optional LLM-driven query expansion.

`Key Functions:`
- load_json: Load JSON files (e.g., topics).
- load_qrel: Parse Qrel TSV files into DataFrames.
- clean_text: Standardizes and cleans input text.
- preprocess_data: Prepares topics, answers, and relevance judgments for further use.

# model_utils.py

- Provides utilities for interacting with LLMs and formatting outputs.

`Key Features:`
- LLM model loading via Hugging Face (Llama 3.1-8B-Instruct used here).
- Query expansion using LLM prompts.
- Pairwise document re-ranking via zero-shot prompts.
- Formatting retrieval results in TREC format.

`Key Functions:`
- load_llama_model: Loads the Llama model.
- generate_response: Generates responses using a given prompt.
- expand_queries_with_llm: Expands input queries.
- pairwise_rerank: Ranks document pairs based on relevance.
- save_results_to_trec_format: Saves results for evaluation tools.

# pairwise_r.py

- Implements pairwise re-ranking of documents using LLMs.
- Processes query-document pairs and ranks them based on zero-shot relevance prompts.

`Key Components:`
- Dynamic indexing of topics.
- Batch processing of document pairs for scalability.
- TREC-compliant result formatting.

# query_exp.py

- Focuses on query expansion using predefined prompts and LLMs.
- Expands queries by including synonyms, related concepts, and contextual information.
`Key Features:`
- Supports batch expansion for efficient processing.
- Integrates expanded queries into downstream retrieval pipelines.

# Approach
1. Data Preprocessing

    Input: Topics (JSON), Qrel (TSV), and optionally document collections.
    Processing:
    Load and clean the input data using load_data_llm.py.
    Index topics for efficient lookup.

2. Query Expansion

    Use LLMs to expand queries for better recall.
    Augment queries with semantically related terms, synonyms, and context.

3. Pairwise Re-ranking

    Compare document pairs for relevance to a query.
    Use zero-shot prompts to evaluate which document is more relevant.
    Rank documents based on pairwise comparisons.

4. Evaluation

    Results are evaluated using metrics like:
    nDCG@k
    P@k (Precision at top-k)
    mAP (Mean Average Precision)
    bpref (Binary Preference).
    Results are saved in TREC format for compatibility with standard tools like trec_eval.

# How to Run
python src/pairwise_exp.py   
    --topics_file files/topics_2.json \   
    --expanded_results_file results/expanded_queries_results_2.tsv \   
    --output_file results/pairwise_expa_results_2.tsv

python src/query_exp.py   
    --topics_file files/topics_2.json \   
    --cross_encoder_results_file results/cross_encoder_2.tsv \  
    --output_file results/expanded_queries_results_2.tsv  \ 
    --index_file data/indexed_topics_2.pkl

python src/pairwise_r.py 
    --topics_file files/topics_2.json \
    --cross_encoder_results_file results/cross_encoder_2.tsv \
    --output_file results/pairwise_results_2.tsv \
    --index_file indexed_topics_2.pkl

python src/evaluate.py 
    --qrel_file files/qrel_1.tsv \
    --run_files results/pairwise_results_1.tsv results/expanded_queries_results_1.tsv results/pairwise_expa_results_1.tsv \
    --model_names "Pairwise Re-ranking" "Query Expansion" "Pairwise on Expanded Queries" \
    --output_prefix results/evaluation

# INDIVIDUAL CONTRIBUTION
### Mucyo's Contribution:
Mucyo was responsible for developing the `evaluate.py` and `pairwise_exp.py` scripts entirely. He implemented evaluation metrics, generated visualizations like ski-jump plots, and designed pairwise re-ranking using LLMs to enhance query-document relevance. Additionally, he authored the `README.md` file and contributed collaboratively to other components of the project.

### Aubin's Contribution:
Aubin handled all other tasks, including developing and refining the remaining scripts (`load_data_llm.py`, `model_utils.py`, `query_exp.py`, `pairwise_r.py`) and ensuring seamless integration between components. His work was central to the project's overall implementation, testing, and optimization.

