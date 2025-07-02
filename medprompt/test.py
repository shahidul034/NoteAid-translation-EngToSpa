import os
import json
import time
import faiss
import numpy as np
from openai import OpenAI
import pickle
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from eval import EvaluateMetric 

# --- Configuration ---

# Choose embedding source: 'local' or 'openai'
EMBEDDING_SOURCE = "local" 

# Local Model Configuration
# LOCAL_EMBEDDING_MODEL_NAME = "abhinand/MedEmbed-small-v0.1"
# EXPECTED_LOCAL_EMBEDDING_DIM = 384
LOCAL_EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  
EXPECTED_LOCAL_EMBEDDING_DIM = 768

# OpenAI Model Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" 
EXPECTED_OPENAI_EMBEDDING_DIM = 1536 

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# --- Retrieval Configuration ---
# How many initial candidates to fetch from each retriever
INITIAL_DENSE_K = 10
INITIAL_SPARSE_K = 10
# Final number of examples for the prompt after re-ranking
FINAL_K = 5 

# Adjust paths as necessary for your environment
TRAINING_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/scielo/es_en_train_aligned.jsonl"
TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/scielo/es_en_test_aligned.jsonl"
# TRAINING_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/TrainingDataMinusOverlaps.json"
# TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
OUTPUT_JSON_PATH = "translated_output.json"

# Derive index/db paths based on embedding source to avoid conflicts
BASE_INDEX_NAME = f"index_{EMBEDDING_SOURCE}"
FAISS_INDEX_PATH = f"{BASE_INDEX_NAME}_faiss.bin"
DATABASE_PATH = f"{BASE_INDEX_NAME}_data.pkl"
BM25_INDEX_PATH = f"{BASE_INDEX_NAME}_bm25.pkl"

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if not client.api_key:
        raise ValueError("OpenAI API key not found in environment variables.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

class MedPromptSystem:
    def __init__(self, embedding_source: str = EMBEDDING_SOURCE):
        self.embedding_source = embedding_source
        self.index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.train_data: List[Tuple[str, str]] = []
        self.tokenized_corpus_for_bm25: Optional[List[List[str]]] = None

        self.embedding_model: Optional[Any] = None 
        self.embedding_dim: Optional[int] = None
        self.openai_client: Optional[OpenAI] = client
        self.cross_encoder: Optional[CrossEncoder] = None

        if self.embedding_source == 'local':
            self.embedding_dim = EXPECTED_LOCAL_EMBEDDING_DIM
            try:
                print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL_NAME}...")
                self.embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
                # Verify dimension
                test_emb = self.embedding_model.encode("test")
                actual_dim = len(test_emb)
                if actual_dim != self.embedding_dim:
                     print(f"Warning: Local model {LOCAL_EMBEDDING_MODEL_NAME} has dimension {actual_dim}, but expected {self.embedding_dim}. Using {actual_dim}.")
                     self.embedding_dim = actual_dim
                print(f"Local embedding model loaded. Dimension: {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading sentence transformer model '{LOCAL_EMBEDDING_MODEL_NAME}': {e}")
                self.embedding_model = None 
                self.embedding_dim = None

        elif self.embedding_source == 'openai':
            self.embedding_dim = EXPECTED_OPENAI_EMBEDDING_DIM
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key (OPENAI_API_KEY) not found in environment variables.")
                self.openai_client = OpenAI(api_key=api_key)
                print(f"OpenAI client initialized for model {OPENAI_EMBEDDING_MODEL}. Expected dimension: {self.embedding_dim}")
                self.embedding_model = OPENAI_EMBEDDING_MODEL 
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.openai_client = None
                self.embedding_model = None
                self.embedding_dim = None
        
        else:
            raise ValueError(f"Invalid embedding_source: '{self.embedding_source}'. Choose 'local' or 'openai'.")

        # Load Cross-Encoder (independent of embedding source)
        try:
            print(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL_NAME}...")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            print("Cross-encoder model loaded successfully.")
        except Exception as e:
            print(f"Error loading cross-encoder model '{CROSS_ENCODER_MODEL_NAME}': {e}")
            self.cross_encoder = None

        # Load indices only if embedding setup was successful
        if self.embedding_dim is not None:
            self.load_indices()
        else:
            print("Skipping index loading due to embedding model initialization failure.")


    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using the configured source (local or OpenAI)."""
        if self.embedding_model is None or self.embedding_dim is None:
            print("Error: Embedding model/client not properly initialized.")
            return None

        text_to_embed = text.strip()
        if not text_to_embed:
            print("Warning: Attempting to embed empty text. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            if self.embedding_source == 'local':
                # Use SentenceTransformer
                embedding = self.embedding_model.encode(text_to_embed, convert_to_numpy=True)
                return embedding.astype(np.float32)

            elif self.embedding_source == 'openai':
                # Use OpenAI API with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.openai_client.embeddings.create(
                            input=text_to_embed,
                            model=self.embedding_model # Use the stored model name
                        )
                        return np.array(response.data[0].embedding, dtype=np.float32)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"OpenAI Embedding Error: {e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"OpenAI Embedding Error: Failed after {max_retries} attempts: {e}")
                            raise # Re-raise the exception after final attempt
            else:
                 # Should not happen if __init__ validation works
                 print(f"Error: Unknown embedding source '{self.embedding_source}' in get_embedding.")
                 return None

        except Exception as e:
            print(f"Error getting embedding for text '{text_to_embed[:50]}...' using {self.embedding_source}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32) # Return zero vector on error

    def save_indices(self) -> None:
        """Save FAISS index, BM25 index, and metadata to disk."""
        # --- Uses FAISS_INDEX_PATH, DATABASE_PATH, BM25_INDEX_PATH which are now source-dependent ---
        if self.index:
            try:
                faiss.write_index(self.index, FAISS_INDEX_PATH)
                print(f"FAISS index ({self.index.ntotal} vectors) saved to {FAISS_INDEX_PATH}.")
            except Exception as e:
                print(f"Error saving FAISS index: {e}")
        else:
            print("Warning: FAISS index not initialized. Cannot save.")

        try:
            with open(DATABASE_PATH, "wb") as f:
                pickle.dump(self.train_data, f)
            print(f"Database ({len(self.train_data)} entries) saved to {DATABASE_PATH}.")
        except Exception as e:
            print(f"Error saving database: {e}")

        if self.bm25_index and self.tokenized_corpus_for_bm25:
            try:
                bm25_data_to_save = {
                    'bm25_index': self.bm25_index,
                    'tokenized_corpus': self.tokenized_corpus_for_bm25
                }
                with open(BM25_INDEX_PATH, "wb") as f:
                    pickle.dump(bm25_data_to_save, f)
                print(f"BM25 index (for {len(self.tokenized_corpus_for_bm25)} docs) saved to {BM25_INDEX_PATH}.")
            except Exception as e:
                print(f"Error saving BM25 index: {e}")
        else:
             print("Warning: BM25 index not initialized or tokenized corpus missing. Cannot save BM25.")

    def load_indices(self) -> None:
        """Load FAISS index, BM25 index, and metadata if available."""
        # --- Uses FAISS_INDEX_PATH, DATABASE_PATH, BM25_INDEX_PATH ---
        faiss_loaded = False
        if not self.embedding_dim:
            print("Error: Cannot load indices without a valid embedding dimension.")
            return

        # Load FAISS and Database
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DATABASE_PATH):
            print(f"Attempting to load FAISS index ({FAISS_INDEX_PATH}) and database ({DATABASE_PATH})...")
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                if self.index.d != self.embedding_dim:
                    print(f"Error: Loaded FAISS index dim ({self.index.d}) != expected model dim ({self.embedding_dim}). Rebuilding needed.")
                    self.index = None
                    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
                    if os.path.exists(DATABASE_PATH): os.remove(DATABASE_PATH)
                else:
                    with open(DATABASE_PATH, "rb") as f:
                        self.train_data = pickle.load(f)
                    print(f"FAISS index loaded ({self.index.ntotal} vectors). Database loaded ({len(self.train_data)} entries).")
                    if self.index.ntotal != len(self.train_data):
                        print(f"Warning: FAISS size ({self.index.ntotal}) != DB size ({len(self.train_data)}). Check consistency.")
                    faiss_loaded = True
            except Exception as e:
                print(f"Error loading FAISS index or database: {e}. Will try to rebuild.")
                self.index = None
                self.train_data = []
        else:
            print(f"Saved FAISS index ({FAISS_INDEX_PATH}) or database ({DATABASE_PATH}) not found.")

        # Load BM25 Index
        if os.path.exists(BM25_INDEX_PATH):
            print(f"Attempting to load BM25 index ({BM25_INDEX_PATH})...")
            try:
                with open(BM25_INDEX_PATH, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['bm25_index']
                    self.tokenized_corpus_for_bm25 = bm25_data['tokenized_corpus']
                if faiss_loaded and len(self.tokenized_corpus_for_bm25) != len(self.train_data):
                     print(f"Warning: Loaded BM25 corpus size ({len(self.tokenized_corpus_for_bm25)}) != DB size ({len(self.train_data)}).")
                     # Optionally invalidate BM25 if mismatch is critical
                print(f"BM25 index loaded (for {len(self.tokenized_corpus_for_bm25)} documents).")
            except Exception as e:
                print(f"Error loading BM25 index: {e}. Will try to rebuild.")
                self.bm25_index = None
                self.tokenized_corpus_for_bm25 = None
        else:
            print(f"Saved BM25 index ({BM25_INDEX_PATH}) not found.")

        if not self.index or not self.bm25_index:
             print("One or more indices not found or failed to load. Will be created during preprocessing if needed.")

    def preprocess(self, training_data: List[Dict[str, str]]) -> None:
        """Generate embeddings (using configured source), build FAISS/BM25 indices, and save."""
        if (self.index and self.bm25_index and self.train_data and self.tokenized_corpus_for_bm25 and
            self.index.ntotal == len(self.train_data) == len(self.tokenized_corpus_for_bm25)):
            print(f"Indices for '{self.embedding_source}' seem loaded and consistent. Skipping preprocessing.")
            return
        if self.embedding_model is None or self.embedding_dim is None:
             print(f"Error: Embedding model/client for '{self.embedding_source}' not loaded. Cannot run preprocessing.")
             return

        print(f"Preprocessing {len(training_data)} examples for '{self.embedding_source}'...")
        self.train_data = []
        embeddings = []
        corpus_indices_for_bm25 = [] # Store original indices of data successfully embedded

        # --- Batching needs to be handled differently for local vs OpenAI ---
        # Local SentenceTransformer handles batching efficiently internally.
        # OpenAI might benefit from batching API calls, but let's keep it simple first
        # by calling get_embedding individually (which handles OpenAI retries).

        print(f"Generating embeddings using '{self.embedding_source}'...")
        for i, entry in enumerate(training_data):
            english_text = entry.get("english")
            spanish_text = entry.get("spanish")
            if not english_text or not spanish_text:
                 print(f"Warning: Skipping entry {i} due to missing English or Spanish text.")
                 continue

            emb = self.get_embedding(english_text) # Use the unified method

            if emb is not None and len(emb) == self.embedding_dim:
                embeddings.append(emb)
                self.train_data.append((english_text, spanish_text)) # Store pair
                corpus_indices_for_bm25.append(i) # Track original index
            else:
                print(f"Warning: Skipping entry {i} due to invalid embedding result (None or wrong dimension).")

            if (i + 1) % 100 == 0: # Print progress
                 print(f"  Processed {i+1}/{len(training_data)} entries for embedding...")

        if not embeddings:
            print("Error: No valid embeddings generated. Cannot build indices.")
            return
        if len(embeddings) != len(self.train_data):
             print(f"CRITICAL ERROR: Mismatch embeddings ({len(embeddings)}) vs stored data ({len(self.train_data)}).")
             return

        # Build FAISS index
        print(f"Building FAISS index with {len(embeddings)} vectors (dim={self.embedding_dim})...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        try:
            embeddings_np = np.array(embeddings).astype(np.float32)
            self.index.add(embeddings_np)
            print(f"Successfully added {self.index.ntotal} vectors to FAISS index.")
            if self.index.ntotal != len(self.train_data):
                 print(f"CRITICAL WARNING: FAISS size ({self.index.ntotal}) != stored data ({len(self.train_data)}).")
        except Exception as e:
            print(f"Error adding embeddings to FAISS index: {e}")
            self.index = None
            return # Stop if FAISS build fails

        # Build BM25 index (on the successfully embedded data)
        print(f"Building BM25 index for {len(self.train_data)} documents...")
        try:
            # Tokenize the English text from the pairs we actually stored
            self.tokenized_corpus_for_bm25 = [pair[0].lower().split() for pair in self.train_data]
            self.bm25_index = BM25Okapi(self.tokenized_corpus_for_bm25)
            print(f"Successfully built BM25 index.")
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            self.bm25_index = None
            # Decide whether to proceed without BM25 or stop

        # Save all successfully built indices
        self.save_indices()

    def knn_retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve kNN examples using FAISS (dense retrieval)."""
        # --- This method now implicitly uses the correct index due to load_indices/preprocess ---
        if self.index is None or self.index.ntotal == 0 or self.embedding_dim is None:
            return []
        if not query: return []

        try:
            query_emb = self.get_embedding(query) # Use the unified embedding function
            if query_emb is None: return []
            query_emb = query_emb.reshape(1, -1)

            # Dimension check should ideally not be needed if initialization is correct, but good safety check
            if query_emb.shape[1] != self.index.d:
                print(f"Error: Query embedding dim ({query_emb.shape[1]}) != index dim ({self.index.d}).")
                return []

            k_to_search = min(k, self.index.ntotal)
            if k_to_search == 0: return []

            distances, indices = self.index.search(query_emb, k_to_search)

            results = []
            if len(indices) > 0:
                for i, idx in enumerate(indices[0]):
                    # idx from FAISS corresponds to the order in self.train_data
                    if 0 <= idx < len(self.train_data):
                        results.append((idx, distances[0][i]))
                    # else: print(f"Warning: Dense retrieved invalid index {idx}.") # Reduce verbosity
            return results
        except Exception as e:
            print(f"Error during FAISS search for query '{query[:50]}...': {e}")
            return []

    def bm25_retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Retrieve examples using BM25 (sparse retrieval)."""
        # --- This method relies on self.bm25_index and self.train_data ---
        if self.bm25_index is None or not self.tokenized_corpus_for_bm25 or not self.train_data:
            return []
        if not query: return []

        try:
            tokenized_query = query.lower().split()
            doc_scores = self.bm25_index.get_scores(tokenized_query)

            # Ensure k is valid
            num_docs = len(self.tokenized_corpus_for_bm25)
            k_to_get = min(k, num_docs)
            if k_to_get <= 0: return []

            # Get indices sorted by score desc. Argpartition is faster for large k.
            # Using argsort for simplicity here.
            top_n_indices = np.argsort(doc_scores)[::-1][:k_to_get]

            results = []
            for idx in top_n_indices:
                 score = doc_scores[idx]
                 # BM25 scores can be <= 0 for non-matches, filter them out
                 if score > 0:
                     # idx from BM25 corresponds to the order in self.train_data
                     if 0 <= idx < len(self.train_data):
                         results.append((idx, score))
                     # else: print(f"Warning: Sparse retrieved invalid index {idx}.") # Reduce verbosity
            return results
        except Exception as e:
             print(f"Error during BM25 search for query '{query[:50]}...': {e}")
             return []

    def rerank_candidates(self, query: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        """Re-ranks candidate examples using a CrossEncoder model."""
        # --- This method relies on self.cross_encoder and self.train_data ---
        if self.cross_encoder is None:
            print("Warning: CrossEncoder not loaded. Skipping re-ranking.")
            return [(idx, 0.0) for idx in candidate_indices] # Return original indices with dummy scores
        if not candidate_indices:
            return []

        pairs = []
        valid_indices_for_reranking = []
        for idx in candidate_indices:
            if 0 <= idx < len(self.train_data):
                english_text, _ = self.train_data[idx]
                pairs.append([query, english_text])
                valid_indices_for_reranking.append(idx)
            # else: print(f"Warning: Invalid index {idx} passed to rerank_candidates.") # Reduce verbosity

        if not pairs:
            return []

        try:
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False) # Set show_progress_bar=True if needed
            scored_candidates = list(zip(valid_indices_for_reranking, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates
        except Exception as e:
            print(f"Error during cross-encoder prediction: {e}")
            return [(idx, 0.0) for idx in valid_indices_for_reranking] # Fallback

    def _format_prompt(self, query: str, examples: List[Tuple[str, str]], direction: str) -> Tuple[str, str]:
        """Helper to format the prompt string and system message."""
        # --- No changes needed here, formatting logic is independent of embedding source ---
        example_texts = [f"Example:\nEnglish: {eng}\nSpanish: {spa}\n" for eng, spa in examples]
        example_section = "\n---\n".join(example_texts)

        if direction == "en_to_es":
            system_prompt = f"You are an expert medical translator (English to Spanish). Use examples for context."
            task_description = f"Translate the following English medical text to Spanish accurately.\n\n## English:\n{query}\n\n## Spanish Translation:"
        else: # es_to_en
             system_prompt = f"You are an expert medical translator (Spanish to English). Use examples for context."
             task_description = f"Translate the following Spanish medical text to English accurately.\n\n## Spanish:\n{query}\n\n## English Translation:"

        prompt_text = f"Examples:\n{example_section}\n\n-----------------------------------------\n\n{task_description}"
        return system_prompt, prompt_text

    def generate_translation(self, query: str, direction: str = "en_to_es") -> str:
        """Generate translation using Hybrid Retrieval, Re-ranking, and LLM generation."""

        # Check if essential components are ready
        if self.index is None or self.bm25_index is None:
             return f"Error: Required indices (FAISS or BM25 for '{self.embedding_source}') not available."
        if not self.cross_encoder:
             print("Warning: Cross-encoder not available, re-ranking will be skipped.")
        if not query:
            return "Error: Input query is empty."

        # Initialize OpenAI client specifically for generation if not already done
        generation_client = self.openai_client # Reuse if source is openai
        if not generation_client:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    return "Error: OpenAI API key needed for generation, but not found."
                generation_client = OpenAI(api_key=api_key)
                print("Initialized separate OpenAI client for generation.")
            except Exception as e:
                 return f"Error: Failed to initialize OpenAI client for generation: {e}"

        # 1. Initial Retrieval
        dense_candidates = self.knn_retrieve(query, k=INITIAL_DENSE_K)
        sparse_candidates = self.bm25_retrieve(query, k=INITIAL_SPARSE_K)

        # # Print all candidates for debugging
        # print(f"Dense candidates: {dense_candidates}")
        # print(f"Sparse candidates: {sparse_candidates}")

        # 2. Combine and Deduplicate
        combined_candidates = {}
        for idx, _ in dense_candidates: combined_candidates[idx] = 1 # Use dict for unique indices
        for idx, _ in sparse_candidates: combined_candidates[idx] = 1
        unique_candidate_indices = list(combined_candidates.keys())

        # 3. Re-rank
        if self.cross_encoder and unique_candidate_indices:
            print(f"Re-ranking {len(unique_candidate_indices)} candidates...") 
            reranked_candidates = self.rerank_candidates(query, unique_candidate_indices)
            final_indices = [idx for idx, score in reranked_candidates[:FINAL_K]]
            print(f"Top {len(final_indices)} indices after re-ranking: {final_indices}") 
        else:
            # print("Skipping re-ranking. Using initial unique indices.") # Reduce verbosity
            final_indices = unique_candidate_indices[:FINAL_K] # Simple fallback

        # 4. Get Final Examples
        final_examples = [self.train_data[idx] for idx in final_indices if 0 <= idx < len(self.train_data)]

        # print(f"Final examples for prompt: {final_examples}")

        # 5. Format Prompt
        system_prompt, prompt_text = self._format_prompt(query, final_examples, direction)

        # 6. Call LLM Generation
        try:
            # Use the dedicated generation client
            response = generation_client.chat.completions.create(
                model="gpt-4o-mini", # Choose appropriate generation model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            translation = response.choices[0].message.content.strip()
            return translation
        except Exception as e:
            print(f"Error generating translation via API for query '{query[:50]}...': {e}")
            return f"Error: Failed to generate translation via API."

    def process_test_set(self, test_data: List[Dict[str, str]], limit: Optional[int] = None) -> None:
        """Process test data (English to Spanish), save results."""
        forward_results = []
        data_to_process = test_data[:limit] if limit is not None else test_data
        total = len(data_to_process)

        print(f"Processing {total} test examples (en_to_es) using '{self.embedding_source}' embeddings...")
        for i, entry in enumerate(data_to_process):
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"Progress: {i+1}/{total}")

            original_english = entry.get("english")
            target_spanish = entry.get("spanish")
            if not original_english or not target_spanish: continue

            try:
                start_time = time.time()
                translated_spanish = self.generate_translation(original_english, "en_to_es")
                end_time = time.time()
                forward_results.append({
                    "id": i, "original_english": original_english, "target_spanish": target_spanish,
                    "translated_spanish": translated_spanish, "latency_seconds": end_time - start_time
                })
            except Exception as e:
                print(f"Error processing test example {i}: {e}")
                forward_results.append({
                    "id": i, "original_english": original_english, "target_spanish": target_spanish,
                    "translated_spanish": f"ERROR: {e}", "latency_seconds": 0
                })

        try:
            # Save with embedding source in filename? Optional.
            output_path = f"{OUTPUT_JSON_PATH.replace('.json', '')}_{self.embedding_source}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(forward_results, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")

def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Loads data from a JSON Lines file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try: data.append(json.loads(line))
                except json.JSONDecodeError: print(f"Warn: Skipping invalid JSON line {i+1} in {file_path}")
        return data
    except FileNotFoundError: print(f"Error: File not found: {file_path}"); return []
    except Exception as e: print(f"Error reading {file_path}: {e}"); return []

# Main execution block
if __name__ == "__main__":
    start_time_main = time.time()
    try:
        print(f"Configured Embedding Source: {EMBEDDING_SOURCE}")
        print("Loading training and test data...")
        # Choose loading method based on file type (JSON vs JSONL)
        train_data = load_jsonl(TRAINING_DATA_PATH)
        test_data = load_jsonl(TEST_DATA_PATH)

        # single_test_data = {
        #     "english": "If you have a weakened immune system due to AIDS, cancer, transplantation, or corticosteroid use, call your doctor if you develop a cough, fever, or shortness of breath.",
        #     "spanish": "Si usted tiene un sistema inmunitario debilitado a causa del SIDA, cáncer, trasplante o uso de corticosteroides, llame al médico si presenta fiebre, tos o dificultad para respirar."
        # }

        # print(f"\n--- Testing with Single Data Point ---")
        # print(f"Input English: {single_test_data['english']}")
        # if 'spanish' in single_test_data:
        #     print(f"Target Spanish (for reference): {single_test_data['spanish']}")
        # print("------------------------------------")

        # with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        #     train_data = json.load(f)
            
        # with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        #     test_data = json.load(f)

        if not train_data or not test_data:
             print("Error: Could not load training or test data. Exiting.")
             exit()
        print(f"Loaded {len(train_data)} training, {len(test_data)} testing examples.")

        # Initialize system with the configured embedding source
        medprompt = MedPromptSystem(embedding_source=EMBEDDING_SOURCE)

        # Preprocess 
        if medprompt.embedding_model is not None:
            medprompt.preprocess(train_data) # Will skip if indices are loaded and valid

            # Check if indices are ready before processing test set
            if medprompt.index and medprompt.bm25_index:
                # --- Call generate_translation directly ---
                # print("\nGenerating translation for the single data point...")
                # start_time_single = time.time()
                # generated_translation = medprompt.generate_translation(
                #     query=single_test_data['english'],
                #     direction="en_to_es" # Assuming English to Spanish
                # )
                # end_time_single = time.time()
                # print("------------------------------------")
                # print(f"Generated Spanish: {generated_translation}")
                # print(f"Translation Time: {end_time_single - start_time_single:.2f} seconds")
                # print("------------------------------------")
                # -------------------------------------------

                TEST_SET_LIMIT = None # Limit for testing, set to None to run all
                medprompt.process_test_set(test_data, limit=TEST_SET_LIMIT)

                output_path_eval = f"{OUTPUT_JSON_PATH.replace('.json', '')}_{medprompt.embedding_source}.json"
                print(f"\nEvaluating results from {output_path_eval}...")
                if os.path.exists(output_path_eval):
                    try:
                        evaluator = EvaluateMetric(output_path_eval, "translated_spanish", "target_spanish", "original_english")
                        print("Evaluation Metrics:")
                        evaluator.evaluate("BLEU")
                        evaluator.evaluate("ROUGE")
                        evaluator.evaluate("BERTSCORE")
                        evaluator.evaluate("COMET")

                    except ImportError:
                         print("EvaluateMetric class not found or import failed. Skipping evaluation.")
                    except Exception as e:
                         print(f"Error during evaluation: {e}")
                else:
                    print(f"Evaluation skipped: Output file not found at {output_path_eval}")
            else:
                print("Skipping test processing and evaluation because indices are not ready.")
        else:
            print("Skipping preprocessing and subsequent steps due to embedding model initialization failure.")

        # output_path_eval = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/translated_output_openai.json"
        # print(f"\nEvaluating results from {output_path_eval}...")
        # if os.path.exists(output_path_eval):
        #     evaluator = EvaluateMetric(output_path_eval, "translated_spanish", "target_spanish", "original_english")
        #     print("Evaluation Metrics:")
        #     evaluator.evaluate("BLEU")
        #     evaluator.evaluate("ROUGE")
        #     evaluator.evaluate("BERTSCORE")
        #     evaluator.evaluate("COMET")


    except Exception as e:
        import traceback
        print(f"\n--- An error occurred during main execution ---")
        print(f"Error Type: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("-------------------------------------------------")
    finally:
         end_time_main = time.time()
         print(f"\nTotal execution time: {end_time_main - start_time_main:.2f} seconds.")
