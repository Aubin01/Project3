import json
import pandas as pd
import re
import pickle
from model_utils import expand_queries_with_llm, load_llama_model

# Loading Functions
def load_json(file_path):
    """Loads a JSON file and returns the data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded JSON data from {file_path} with {len(data)} entries.")
    return data

def load_qrel(file_path):
    """Loads the QREL file and returns it as a DataFrame."""
    df = pd.read_csv(file_path, sep='\t', header=None, names=["query_id", "iter", "doc_id", "relevance"])
    print(f"Loaded QREL data from {file_path} with {df.shape[0]} entries.")
    return df

# Cleaning Function
def clean_text(text):
    """Cleans text by removing HTML tags, URLs, and normalizing whitespace."""
    clean_text = re.sub(r'<[^>]+>', '', text)
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
    return clean_text

# Preprocessing and Indexing Functions
def preprocess_data(topics_file, answers_file=None, qrel_file=None, use_expansion=False):
    """
    Preprocesses data from topics, answers, and qrel files. Cleans the text and optionally applies query expansion.
    """
    topics = load_json(topics_file)
    if answers_file:
        answers = load_json(answers_file)
    else:
        answers = None
    qrel = load_qrel(qrel_file) if qrel_file else None

    # Clean text in topics (e.g., title field)
    for topic in topics:
        if "Title" in topic:
            topic["Title"] = clean_text(topic["Title"])

    if use_expansion:
        print("Loading LLM for query expansion...")
        llm_model = load_llama_model()
        prompt_template = "Expand the following query with related terms:\nQuery: {query}\nExpanded Terms:"
        expanded_topics = expand_queries_with_llm({str(t["Id"]): t["Title"] for t in topics}, llm_model, prompt_template)
        
        # Add expanded queries back to the topics
        for topic in topics:
            topic["ExpandedTitle"] = expanded_topics.get(str(topic["Id"]), topic["Title"])
        print("Query expansion completed.")

    print(f"Preprocessed topics with {len(topics)} entries.")
    return topics, answers, qrel

def index_queries(topics):
    """Organizes topics by query ID and creates an indexed dictionary for faster access."""
    indexed_topics = {str(topic["Id"]): topic for topic in topics}
    print(f"Indexed {len(indexed_topics)} queries by their IDs.")
    return indexed_topics

# Save and Load Indexed Data
def save_index(data, file_path):
    """Saves indexed data as a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Index saved to {file_path}")

def load_index(file_path):
    """Loads indexed data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded index from {file_path}")
    return data
