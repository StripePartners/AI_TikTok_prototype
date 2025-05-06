import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import TokenTextSplitter
from sklearn.metrics import f1_score
import torch
from typing import List

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
    JACCARD_DISTANCE_THRESHOLD = 1.0 - similarity_threshold

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

def calculate_pairwise_cosine_similarities(responses: List[str]) -> float:
    """
    Calculate pairwise cosine similarities between a list of responses.
    """
    embeddings = model.encode(responses, convert_to_tensor=True, device=device)
    cosine_similarities = util.cos_sim(embeddings, embeddings)
    cosine_similarities = cosine_similarities.cpu().numpy()
    print(f"Cosine Similarities Matrix:\n{cosine_similarities}")

    triu_indices = np.triu_indices(cosine_similarities.shape[0], k=1)
    upper_triangle_values = cosine_similarities[triu_indices]
    return upper_triangle_values.mean()

def calculate_pairwise_jaccard_similarities(responses: List[str]) -> float:
    """
    Calculate pairwise Jaccard similarities between a list of responses.
    """
    jaccard_similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            set1 = set(responses[i].lower().split())
            set2 = set(responses[j].lower().split())
            jaccard_sim = 1 - _calculate_jaccard_distance(set1, set2)
            jaccard_similarities.append(1 - jaccard_sim)
    print(f"Jaccard Similarities: {jaccard_similarities}")
    return np.mean(jaccard_similarities)

# Load the SentenceTransformer model
model_name = 'all-MiniLM-L6-v2'
try:
    model = SentenceTransformer(model_name)
except Exception as e:
    print(f"Error loading Sentence Transformer model '{model_name}': {e}")
    print("Please ensure the model name is correct and you have an internet connection.")
    exit()

# Check accuracy of factual probe results
def evaluate_factual_probe_results(file_path):
    df = pd.read_csv(file_path)
    df['correct'] = df.apply(lambda row: hard_match(row['gold_answer'], row['llm_response']), axis=1)
    df['jacccard'] = df.apply(lambda row: is_concept_in_sentence(row['gold_answer'], row['llm_response']), axis=1)
    df['cosine'] = df.apply(lambda row: is_concept_in_sentence_cosine(row['gold_answer'], row['llm_response']), axis=1)

    # print(df.head())

    percentage_correct = (df['correct'].sum() / len(df)) * 100
    percentage_correct_jaccard = (df['jacccard'].sum() / len(df)) * 100
    percentage_correct_cosine = (df['cosine'].sum() / len(df)) * 100

    df.to_csv('factual_probe_results/test.csv', index=False)

    print(f"Percentage of correct rows: hard match: {percentage_correct:.2f}%")
    print(f"Percentage of correct rows: Jaccard match: {percentage_correct_jaccard:.2f}%")
    print(f"Percentage of correct rows: Cosine match: {percentage_correct_cosine:.2f}%")

    return df

# Check similarities across responses for each model - approach 1
def evaluate_consistency_1(file_path):
    df = pd.read_excel(file_path, skiprows=1)
    model_types = df['model'].unique()
    include_context_list = df['kb_included'].unique()
    video_indices = df['video_index'].unique()
    results = []
    for model_type in model_types:
        for include_context in include_context_list:
            for video_index in video_indices:
                # Get all responses for this model, context, and video index
                responses = df[(df['model'] == model_type) & (df['kb_included'] == include_context) & (df['video_index'] == video_index)]['response'].tolist()
                if len(responses) > 1:  # Ensure there are at least two responses to compare
                    avg_jaccard_similarity = calculate_pairwise_jaccard_similarities(responses)
                    avg_cos_similarity = calculate_pairwise_cosine_similarities(responses)
                    print(f"Model: {model_type}, Context: {include_context}, Video Index: {video_index}, Avg Cosine Similarity: {avg_cos_similarity:.4f}, Avg Jaccard Similarity: {avg_jaccard_similarity:.4f}")
                    results.append({
                        'model': model_type,
                        'kb_included': include_context,
                        'video_index': video_index,
                        'avg_cosine_similarity': avg_cos_similarity,
                        'avg_jaccard_similarity': avg_jaccard_similarity
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("consistency_eval_similarities_1.csv", index=False)
    # Calculate the average similarity for each model and context
    avg_similarities = results_df.groupby(
        ['model', 'kb_included'])[['avg_cosine_similarity', 'avg_jaccard_similarity']].mean().reset_index()
    print(avg_similarities)
    return results_df

# Check similarities across responses for each model - approach 2
def evaluate_consistency_2(file_path):
    df = pd.read_excel(file_path, skiprows=1)
    model_types = df['model'].unique()
    include_context_list = df['kb_included'].unique()
    video_indices = df['video_index'].unique()
    results = []
    for model_type in model_types:
        for include_context in include_context_list:
                # Get all responses for this model and context (regardless of video)
                responses = df[(df['model'] == model_type) & (df['kb_included'] == include_context)]['response'].tolist()
                if len(responses) > 1:  # Ensure there are at least two responses to compare
                    avg_jaccard_similarity = calculate_pairwise_jaccard_similarities(responses)
                    avg_cos_similarity = calculate_pairwise_cosine_similarities(responses)
                    print(f"Model: {model_type}, Context: {include_context}, Avg Cosine Similarity: {avg_cos_similarity:.4f}, Avg Jaccard Similarity: {avg_jaccard_similarity:.4f}")
                    results.append({
                        'model': model_type,
                        'kb_included': include_context,
                        'avg_cosine_similarity': avg_cos_similarity,
                        'avg_jaccard_similarity': avg_jaccard_similarity
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("consistency_eval_similarities_2.csv", index=False)
    return results_df

# Check accuracy of strategy identification
def calculate_strategy_accuracy(file_path):
    df = pd.read_csv(file_path)
    model_types = df['model'].unique()
    include_context_list = df['kb_included'].unique()
    actual_cols = ['fomo_actual', 'auth_actual', 'overconf_actual']
    predicted_cols = ['fomo_predicted', 'auth_predicted', 'overconf_predicted']
    results = []

    for model_type in model_types:
        for include_context in include_context_list:
            subset = df[(df['model'] == model_type) & (df['kb_included'] == include_context)]
            y_true = subset[actual_cols].values
            y_pred = subset[predicted_cols].values

            print(f"Model: {model_type}, Context: {include_context}")
            print(y_true)
            print(y_pred)

            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            exact_matches = np.all(y_true == y_pred, axis=1)
            exact_match_score = np.mean(exact_matches)

            print(f"Model: {model_type}, Context: {include_context}, F1 Score (macro): {f1_macro:.4f}, F1 Score (micro): {f1_micro:.4f}")
            results.append({
                'model': model_type,
                'kb_included': include_context,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'exact_match_score': exact_match_score
                })
    results_df = pd.DataFrame(results)
    results_df.to_csv("strategy_scores.csv", index=False)
    return results_df


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

# results_df = evaluate_factual_probe_results('factual_probe_results/gpt_4o_scores.csv')
# results_df = evaluate_consistency_1('consistency_eval_results.xlsx')
# results_df = evaluate_consistency_2('consistency_eval_results.xlsx')
results_df = calculate_strategy_accuracy('strategy_eval_results.csv')

print(results_df.head())

# # Testing
# s1 = 'The quick brown fox jumps over the lazy dog'
# s2 = 'The quick brown fox jumps over the lazy dog and runs away'
# # s1 = "intrinsic value"
# # s2 = "According to Warren Buffett's principles, the key measurement used in acquisitions of businesses and common stocks is understanding the business itself. He emphasizes that this means having a reasonably good idea of what the business will look like in five or ten years from an economic standpoint. In other words, the focus is on evaluating the intrinsic value of the business, rather than its market price."
# print(calculate_similarities(s1, s2))