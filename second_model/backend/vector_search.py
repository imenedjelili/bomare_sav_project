# backend/vector_search.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os 

# --- Initialization ---
MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", 'all-MiniLM-L6-v2') # Allow override via env var
log = logging.getLogger(__name__)

log.info(f"VECTOR_SEARCH: Attempting to load sentence transformer model: {MODEL_NAME}")
try:
    model = SentenceTransformer(MODEL_NAME)
    log.info(f"VECTOR_SEARCH: Successfully loaded sentence transformer model: {MODEL_NAME}")
except Exception as e:
    log.error(f"VECTOR_SEARCH: CRITICAL - Failed to load Sentence Transformer model '{MODEL_NAME}': {e}", exc_info=True)
    # This is a critical failure; the application cannot perform RAG without it.
    # Consider raising a more specific custom exception or handling it in the main app startup.
    raise RuntimeError(f"Failed to initialize sentence transformer model: {MODEL_NAME}. RAG will not function.") from e

def load_data(json_path: str) -> list | None:
    """Loads troubleshooting data from a JSON file and flattens it."""
    log.debug(f"VECTOR_SEARCH: Attempting to load and flatten data from {json_path}")
    try:
        with open(json_path, "r", encoding='utf-8') as f:
            raw_data = json.load(f) 
        if not isinstance(raw_data, list):
            log.error(f"VECTOR_SEARCH: Error - Data in {json_path} is not a valid JSON list of TV model entries.")
            return None

        flattened_data = []
        for model_entry_index, model_entry in enumerate(raw_data):
            if not isinstance(model_entry, dict):
                log.warning(f"VECTOR_SEARCH: Skipping non-dict item at index {model_entry_index} in raw_data: {str(model_entry)[:100]}")
                continue

            model_name = model_entry.get("model")
            troubleshooting_issues = model_entry.get("troubleshooting_issues")
            model_general_images = model_entry.get("images") # For associating general images with issues from this model

            if not model_name or not isinstance(model_name, str) or not model_name.strip():
                log.warning(f"VECTOR_SEARCH: Skipping model entry at index {model_entry_index} due to missing or invalid 'model' field: {str(model_entry)[:100]}")
                continue
            model_name = model_name.strip() # Ensure no leading/trailing spaces

            if not isinstance(troubleshooting_issues, list):
                # Log if a model entry has no issues, but don't necessarily fail the whole process
                log.debug(f"VECTOR_SEARCH: Model '{model_name}' (entry index {model_entry_index}) has no 'troubleshooting_issues' list or it's invalid. No issues from this entry will be indexed.")
                continue # Continue to the next model entry

            for issue_index, issue_detail in enumerate(troubleshooting_issues):
                if not isinstance(issue_detail, dict):
                    log.warning(f"VECTOR_SEARCH: Skipping non-dict item in troubleshooting_issues for model '{model_name}' (issue index {issue_index}): {str(issue_detail)[:100]}")
                    continue

                issue_text = issue_detail.get("issue")
                steps = issue_detail.get("steps")

                if not issue_text or not isinstance(issue_text, str) or not issue_text.strip():
                    log.warning(f"VECTOR_SEARCH: Skipping issue for model '{model_name}' (issue index {issue_index}) due to missing, empty, or invalid 'issue' text: {str(issue_detail)[:100]}")
                    continue

                flattened_entry = {
                    "model": model_name, # Store the original model name
                    "issue": issue_text.strip(),
                    "steps": steps if isinstance(steps, list) else [],
                    "images": model_general_images if isinstance(model_general_images, dict) else None # Associate general model images
                    # Add original_model_entry_index and original_issue_index if needed for deep debugging
                    # "original_model_entry_index": model_entry_index,
                    # "original_issue_index": issue_index
                }
                flattened_data.append(flattened_entry)

        if not flattened_data:
            log.error(f"VECTOR_SEARCH: No valid issues to index were extracted and flattened from {json_path}. RAG will be ineffective.")
            return None

        log.info(f"VECTOR_SEARCH: Successfully loaded and flattened {len(flattened_data)} issue items from {len(raw_data)} model entries in {json_path}")
        return flattened_data
    except FileNotFoundError:
        log.error(f"VECTOR_SEARCH: Error - Data file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"VECTOR_SEARCH: Error - Could not decode JSON from {json_path}. Error: {e}")
        return None
    except Exception as e:
        log.error(f"VECTOR_SEARCH: An unexpected error occurred loading and flattening data from {json_path}: {e}", exc_info=True)
        return None

def create_faiss_index(data: list) -> tuple[faiss.Index | None, list | None]: 
    """Creates a FAISS index for the 'issue' field in the (flattened) data."""
    if not data:
        log.error("VECTOR_SEARCH: Cannot create FAISS index from empty or invalid (flattened) data.")
        return None, None

    texts_to_embed = []
    # text_to_original_data_idx_map maps the index in FAISS (and texts_to_embed)
    # to the index in the *flattened data* list.
    text_to_original_data_idx_map = [] 

    log.info("VECTOR_SEARCH: Preparing 'issue' texts from flattened data for embedding...")
    for i, item in enumerate(data): 
        if isinstance(item, dict) and "issue" in item and isinstance(item["issue"], str) and item["issue"].strip():
            issue_text = item["issue"].strip()
            texts_to_embed.append(issue_text) 
            text_to_original_data_idx_map.append(i) 
            log.debug(f"VECTOR_SEARCH_INDEXING: Adding to embed (FlatDataIdx {i}): Model='{item.get('model', 'N/A')}', Issue='{issue_text[:100]}...'")
        else:
            log.warning(f"VECTOR_SEARCH_INDEXING: Flattened item at index {i} is invalid or missing 'issue' field. Skipping. Item: {str(item)[:100]}")

    if not texts_to_embed:
        log.error("VECTOR_SEARCH: No valid 'issue' fields found in flattened data to index. FAISS index will be empty.")
        return None, None

    log.info(f"VECTOR_SEARCH: Found {len(texts_to_embed)} valid issues from flattened data to index.")
    try:
        log.info(f"VECTOR_SEARCH: Encoding {len(texts_to_embed)} issues using '{MODEL_NAME}'...")
        # Consider adding batch_size for very large datasets, e.g., model.encode(..., batch_size=128)
        embeddings = model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True) 
        if embeddings is None or embeddings.size == 0:
            log.error("VECTOR_SEARCH: Encoding resulted in empty embeddings array.")
            return None, None
        
        embeddings_float32 = embeddings.astype('float32') # FAISS typically expects float32
        dimension = embeddings_float32.shape[1]
        log.info(f"VECTOR_SEARCH: Embeddings created with dimension: {dimension}")
        
        index = faiss.IndexFlatL2(dimension) # Using L2 distance (Euclidean)
        # For larger datasets, consider more advanced FAISS indexes like IndexIVFFlat for speed,
        # but IndexFlatL2 is exact and good for moderate sizes.
        index.add(embeddings_float32)
        log.info(f"VECTOR_SEARCH: FAISS index created successfully with {index.ntotal} vectors.")
        return index, text_to_original_data_idx_map
    except Exception as e:
        log.error(f"VECTOR_SEARCH: Error creating FAISS index: {e}", exc_info=True)
        return None, None

def search_relevant_guides(
    query_text: str, 
    target_model: str, 
    data: list,  # This is the FLATTENED data list
    index: faiss.Index, 
    text_to_original_data_idx_map: list, 
    k_results: int = 5 # Retrieve a few top semantic matches to filter by model
) -> dict | None:
    """
    Searches for relevant guides from the FLATTENED data,
    STRICTLY matching the target_model from the top k_results semantic matches.
    Returns a single best guide dictionary (from flattened_data) or None.
    """
    if not all([index, data, text_to_original_data_idx_map, query_text, target_model]):
        log.error("VECTOR_SEARCH: Search cannot be performed - missing critical inputs (index, data, map, query, or target_model).")
        return None
    if index.ntotal == 0:
        log.warning("VECTOR_SEARCH: FAISS index is empty. Cannot perform search.")
        return None
    if not query_text.strip():
        log.warning("VECTOR_SEARCH: Empty query_text provided. Cannot perform search.")
        return None
    
    log.debug(f"VECTOR_SEARCH: Starting search. Query='{query_text}', TargetModel='{target_model}', k_to_retrieve_semantically={k_results}")

    try:
        log.debug(f"VECTOR_SEARCH: Encoding search query: '{query_text[:100]}'")
        query_embedding = model.encode([query_text.strip()], convert_to_numpy=True)
        if query_embedding is None or query_embedding.size == 0:
             log.error("VECTOR_SEARCH: Failed to encode search query (resulted in empty embedding).")
             return None
        
        query_embedding_np = query_embedding.astype('float32')
        
        # Ensure k_results is not greater than the total number of items in the index
        effective_k_semantic = min(k_results, index.ntotal)
        if effective_k_semantic == 0:
            log.warning(f"VECTOR_SEARCH: effective_k_semantic is 0 (k_results={k_results}, index.ntotal={index.ntotal}). Cannot search.")
            return None
            
        log.debug(f"VECTOR_SEARCH: Searching FAISS index for top {effective_k_semantic} semantic matches.")
        # D: distances (L2 squared), I: indices in the FAISS index
        distances, faiss_indices = index.search(query_embedding_np, k=effective_k_semantic)
        
        log.debug(f"VECTOR_SEARCH: Raw FAISS results - Distances: {distances[0]}, FAISS Indices: {faiss_indices[0]}")

        best_match_for_model: dict | None = None
        best_score_for_model = float('inf') # Lower L2 distance is better

        target_model_processed = target_model.strip().lower() # Normalize target model once

        log.debug(f"VECTOR_SEARCH: Evaluating top {effective_k_semantic} semantic candidates against target model '{target_model_processed}':")
        
        for i in range(effective_k_semantic):
            faiss_idx = faiss_indices[0][i]
            score = float(distances[0][i]) # L2 distance

            if not (0 <= faiss_idx < len(text_to_original_data_idx_map)):
                 log.warning(f"VECTOR_SEARCH: Invalid faiss_idx {faiss_idx} from FAISS search (Rank {i+1}). Skipping.")
                 continue
            
            original_data_idx = text_to_original_data_idx_map[faiss_idx] # Map to index in flattened `data`
            
            if not (0 <= original_data_idx < len(data)):
                 log.warning(f"VECTOR_SEARCH: Mapped original_data_idx {original_data_idx} (from faiss_idx {faiss_idx}) is out of bounds "
                             f"for flattened data (len={len(data)}) (Rank {i+1}). Skipping.")
                 continue

            candidate_guide = data[original_data_idx] 
            
            if not isinstance(candidate_guide, dict):
                log.warning(f"VECTOR_SEARCH: Data item at original_data_idx {original_data_idx} is not a dictionary (Rank {i+1}). Skipping.")
                continue

            candidate_model_original_case = candidate_guide.get("model", "")
            candidate_model_processed = candidate_model_original_case.strip().lower()
            candidate_issue = candidate_guide.get("issue","N/A")

            log.debug(f"  - Candidate (Rank {i+1}, FAISS Idx: {faiss_idx}, FlatData Idx: {original_data_idx}, Score: {score:.4f}): "
                      f"Model='{candidate_model_original_case}' (processed: '{candidate_model_processed}'), Issue='{candidate_issue[:70]}...'")

            if candidate_model_processed == target_model_processed:
                log.info(f"VECTOR_SEARCH:   >>> Candidate matches target model '{target_model_processed}'.")
                # If we only want the absolute best semantic match for this model, we can take the first one.
                # If multiple issues for the same model could match semantically, this first one is the most similar.
                if score < best_score_for_model: # Keep track if we wanted to find the one with lowest score among matches
                    best_score_for_model = score
                    best_match_for_model = candidate_guide
                    log.info(f"VECTOR_SEARCH:     This is currently the best match for model '{target_model_processed}' with score {score:.4f}.")
                # For your use case (k_results=1 in handler), this loop will likely only process one if model matches.
                # If k_results in handler is >1, this will find the first semantic match that also matches model.
                # If you want the BEST SEMANTIC match out of ALL candidates for that model, this logic is fine.
        
        if best_match_for_model:
             log.info(f"VECTOR_SEARCH: *** 최종 매치 발견! *** 최종 선택된 가이드: Model='{best_match_for_model.get('model')}', "
                      f"Issue='{best_match_for_model.get('issue', 'N/A')}', Score: {best_score_for_model:.4f}")
        else:
             log.warning(f"VECTOR_SEARCH: No guide strictly matching target model '{target_model_processed}' found within the top "
                         f"{effective_k_semantic} semantic matches for query '{query_text[:60]}...'.")

        return best_match_for_model

    except faiss.FaissException as e_faiss: 
        log.error(f"VECTOR_SEARCH: FAISS Error during search for query '{query_text[:60]}...': {e_faiss}", exc_info=True)
        return None
    except Exception as e_generic:
        log.error(f"VECTOR_SEARCH: Unexpected error during search for query '{query_text[:60]}...': {e_generic}", exc_info=True)
        return None