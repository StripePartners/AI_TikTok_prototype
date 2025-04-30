import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import TokenTextSplitter
import torch

# check whether gold-truth answer is contained in the LLM response (start with hard matching)
def hard_match(gold_answer, llm_response):
    return gold_answer.lower() in llm_response.lower()

# extend to soft matching
def _calculate_jaccard_distance(set1: set, set2: set) -> float:
    """Calculate Jaccard distance between two sets of words"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    # Avoid division by zero
    return 1.0 - (float(intersection) / float(union) if union > 0 else 0.0)

def is_concept_in_sentence(target_answer: str, llm_response: str, similarity_threshold: float = 0.4) -> bool:
    # Check for simple words first (faster) - this check is now redundant here
    # as it's done before calling this function, but kept for completeness
    # if any(token.text.lower() in words_set for token in doc):
    #     return True

    # Tokenise the input strings into words
    target_tokens = set(target_answer.lower().split())
    llm_tokens = llm_response.lower().split()

    # Construct n-grams (4-grams specifically)
    text_ngrams_sets = set()
    # Use n=4 based on the original code range (j = i + 4)
    ngram_size = 4
    if len(llm_tokens) >= ngram_size:
        for i in range(len(llm_tokens) - ngram_size + 1):
            # Create n-gram string and split into a set of words
            ngram = frozenset(llm_tokens[i:i + ngram_size])
            text_ngrams_sets.add(ngram) # Use frozenset for adding to set

    # Check for similar phrases using Jaccard distance
    # Let's add 5-grams as well for closer replication.
    ngram_size = 5
    if len(llm_tokens) >= ngram_size:
        for i in range(len(llm_tokens) - ngram_size + 1):
            ngram = frozenset(llm_tokens[i:i + ngram_size])
            text_ngrams_sets.add(ngram)

    # SIMILARITY_THRESHOLD = 0.6 # Jaccard Similarity = 1 - Jaccard Distance
    JACCARD_DISTANCE_THRESHOLD = 1.0 - similarity_threshold # = 0.4

    for gram_set_froz in text_ngrams_sets:
        # Calculate Jaccard distance (more efficient than similarity)
        # No need to recalculate gram_words set if we store sets
        # print(f"Checking: Target={target_tokens}, Gram={gram_set_froz}")
        dist = _calculate_jaccard_distance(target_tokens, gram_set_froz)
        if dist <= JACCARD_DISTANCE_THRESHOLD:
            # print(f"Match Found: Target={target_tokens}, Gram={gram_set_froz}, Dist={dist}")
            return True

    return False

def _calculate_cosine_similarity(target: str, llm_response: str) -> float:
    """
    Calculates cosine similarities between target and llm response.
    """
    # Encode target response
    target_embedding = model.encode(
        target,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True
    )
    target_embedding_cpu = target_embedding.cpu() # Move to CPU for calculations

    # Encode llm response
    llm_response_embedding = model.encode(
        llm_response,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True
    )
    llm_response_embedding_cpu = llm_response_embedding.cpu() # Move to CPU for calculations

    # Calculate cosine similarity
    cos_score = util.cos_sim(target_embedding_cpu, llm_response_embedding_cpu).item()
    # print(f"Cosine Similarity: {cos_score:.4f}")
    # if cos_score >= threshold:
    #     # print("Match found based on cosine similarity.")
    #     return True

    # return False
    return cos_score

def is_concept_in_sentence_cosine(target_answer: str, llm_response: str, similarity_threshold: float = 0.4) -> bool:
    text_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = text_splitter.split_text(llm_response)

    for chunk in chunks:
        cos_score = _calculate_cosine_similarity(target_answer, chunk)
        if cos_score >= similarity_threshold:
            return True

    return False

# Load the SentenceTransformer model
model_name = 'all-MiniLM-L6-v2'
try:
    model = SentenceTransformer(model_name)
except Exception as e:
    print(f"Error loading Sentence Transformer model '{model_name}': {e}")
    print("Please ensure the model name is correct and you have an internet connection.")
    exit()

# Set device - now with MPS support for Apple Silicon
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA device found. Using GPU.")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # Check if MPS is available and functional
    try:
        # Test MPS with a small tensor operation
        test_tensor = torch.tensor([1.0, 2.0]).to('mps')
        _ = test_tensor * 2
        device = 'mps'
        print("MPS device found and functional. Using Apple Silicon GPU.")
    except Exception as e:
        print(f"MPS device found but encountered an error during test: {e}")
        print("Falling back to CPU.")
        device = 'cpu'
else:
    print("No GPU or MPS found. Using CPU.")

print(f"Using device: {device}")
model = model.to(device)

df = pd.read_csv('factual_probe_results_gpt_4o_scores.csv')
df['correct'] = df.apply(lambda row: hard_match(row['gold_answer'], row['llm_response']), axis=1)
df['jacccard'] = df.apply(lambda row: is_concept_in_sentence(row['gold_answer'], row['llm_response']), axis=1)
df['cosine'] = df.apply(lambda row: is_concept_in_sentence_cosine(row['gold_answer'], row['llm_response']), axis=1)

# print(df.head())

percentage_correct = (df['correct'].sum() / len(df)) * 100
percentage_correct_jaccard = (df['jacccard'].sum() / len(df)) * 100
percentage_correct_cosine = (df['cosine'].sum() / len(df)) * 100

df.to_csv('test.csv', index=False)

print(f"Percentage of correct rows: hard match: {percentage_correct:.2f}%")
print(f"Percentage of correct rows: Jaccard match: {percentage_correct_jaccard:.2f}%")
print(f"Percentage of correct rows: Cosine match: {percentage_correct_cosine:.2f}%")

# # Testing
# s1 = 'The quick brown fox jumps over the lazy dog'
# s2 = 'The quick brown fox jumps over the lazy dog and runs away'
# # s1 = "intrinsic value"
# # s2 = "According to Warren Buffett's principles, the key measurement used in acquisitions of businesses and common stocks is understanding the business itself. He emphasizes that this means having a reasonably good idea of what the business will look like in five or ten years from an economic standpoint. In other words, the focus is on evaluating the intrinsic value of the business, rather than its market price."
# print(calculate_similarities(s1, s2))