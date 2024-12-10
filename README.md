#### Evaluation of Query Expansion and Pairwise Re-ranking Techniques using LLMs

## Project Description
This project focuses on improving information retrieval precision by integrating advanced techniques such as Query Expansion and Pairwise Re-ranking using LLMs. The workflow begins with Cross-Encoder results from previous assignments, which are used as initial candidates for the model. Query Expansion enriches the original queries with contextual terms, while Pairwise Re-ranking ensures the highest relevance by reordering documents based on pairwise comparisons. Additionally, fused results combining Query Expansion and Pairwise Re-ranking outputs were evaluated to explore combined benefits.

## Table of Contents
1. Files
2. Installation
3. How to Run the Code
4. Outputs
5. Evaluation
6. Performance Notes
7. Individual Contributions 

## Files
1. load_data_llm.py:Handles data preprocessing, including query cleaning, indexing, and optional query expansion using LLMs.

2. pairwise_r.py: Implements pairwise re-ranking for Cross-Encoder results using LLMs for enhanced document ranking.

3. query_exp.py:  Expands queries using LLM and evaluates retrieval performance based on the expanded queries.

4. pairwise_exp.py: Combines Pairwise Re-ranking with Query Expansion for a hybrid approach to document ranking.

5. model_utils.py: Contains utility functions for loading the LLaMA model(Llama 3.1-8B-Instruct), setting up the environment, and generating responses 

6. evaluate.py: Calculates retrieval metrics (Precision@1, Precision@5, Precision@10, nDCG@5, MRR, MAP) for all retrieval models and fused results. Also generates ski-jump plots for visual analysis.

## Installation
Prerequisites:

Python 3.x
 Required Python packages: pandas, argparse, matplotlib, ranx, sentence_transformers, toch 

## How to Run the Code

Step 1: Perform Pairwise Re-ranking on Cross-Encoder Results
```bash
python src/pairwise_exp.py   
    --topics_file files/topics_2.json \   
    --expanded_results_file results/expanded_queries_results_2.tsv \   
    --output_file results/pairwise_expa_results_2.tsv
```
Step 2: Perform Query Expansion
```bash
python src/query_exp.py   
    --topics_file files/topics_2.json \   
    --cross_encoder_results_file results/cross_encoder_2.tsv \  
    --output_file results/expanded_queries_results_2.tsv  \ 
    --index_file data/indexed_topics_2.pkl
```
Step 3: Combine Pairwise Re-ranking with Query Expansion
```bash
python src/pairwise_r.py 
    --topics_file files/topics_2.json \
    --cross_encoder_results_file results/cross_encoder_2.tsv \
    --output_file results/pairwise_results_2.tsv \
    --index_file indexed_topics_2.pkl
```
Step 4: Fuse and Evaluate Results
```bash
python src/evaluate.py 
    --qrel_file files/qrel_1.tsv \
    --run_files results/pairwise_results_1.tsv results/expanded_queries_results_1.tsv results/pairwise_expa_results_1.tsv \
    --model_names "Pairwise Re-ranking" "Query Expansion" "Pairwise on Expanded Queries" \
    --output_prefix results/evaluation
```

## Outputs
Generated output files include:

1. Query Expansion Results: expanded_queries_results_2.tsv
2. Pairwise Re-ranking Results: pairwise_results_2.tsv
3. Pairwise on Expanded Queries Results: pairwise_expa_results_2.tsv
4. Fused Results: fused_results_2.tsv

## Evaluation
Evaluate Retrieval Performance:

Upon running the evaluate.py script as outlined in the previous step, you will obtain detailed performance metrics, including Precision@1, Precision@5, Precision@10, Recall, MAP (Mean Average Precision), and nDCG (Normalized Discounted Cumulative Gain) 


## Performance Notes
Execution time varies based on the dataset and retrieval method:

Query Expansion: ~10 minutes for 100 queries
Pairwise Re-ranking: ~20 minutes due to pairwise document comparisons
Fused Results Evaluation: ~2 minutes.


## Individual Contribution

### Mucyo's Contribution:
Mucyo developed the evaluate.py and pairwise_exp.py scripts, implementing evaluation metrics, generating ski-jump plots, and optimizing pairwise re-ranking with LLMs. He also authored the README file.

### Aubin's Contribution:
Aubin developed and refined core scripts (load_data_llm.py, model_utils.py, query_exp.py and pairwise_r.py)

