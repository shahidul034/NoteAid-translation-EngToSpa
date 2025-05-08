# retrieval.py
import random
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import os
import pickle
from tqdm import tqdm

class Retriever:
    def __init__(self, train_data, embedder=None, bm25_tokenized=None,
                 embedding_source='local', openai_client=None, embedding_model_name=None, # Changed from embedding_model
                 index_dir="indices", force_rebuild=False,
                 dataset_name="unknown_dataset", embedder_identifier="unknown_embedder"): # New parameters

        self.train_data_source = train_data
        self.embedder = embedder
        self.embedding_dim = None
        
        # Try to set embedding_dim
        if self.embedder and hasattr(self.embedder, 'get_sentence_embedding_dimension'):
             self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        elif self.embedder and embedding_source == 'local' and len(train_data) > 0 and train_data[0] and len(train_data[0]) > 0:
            try:
                sample_emb = self._get_embedding(train_data[0][0]) # Assuming train_data[0][0] is a valid text string
                self.embedding_dim = sample_emb.shape[0]
            except Exception as e:
                print(f"Warning: Could not infer embedding dimension for local model: {e}")
        elif embedding_source == 'openai' and embedding_model_name:
            # Common OpenAI model dimensions
            if "ada-002" in embedding_model_name or "3-small" in embedding_model_name:
                self.embedding_dim = 1536
            elif "3-large" in embedding_model_name:
                self.embedding_dim = 3072
            else:
                print(f"Warning: Unknown OpenAI embedding model '{embedding_model_name}' for dimension setting. Please set manually or update list.")

        self.embedding_source = embedding_source
        self.openai_client = openai_client
        self.embedding_model_name = embedding_model_name # Storing the actual model name string

        self.index = None
        self.train_data_for_faiss = []
        self.bm25_index = None
        self.tokenized_corpus_for_bm25 = bm25_tokenized

        # Define paths for indices using dataset_name and embedder_identifier
        os.makedirs(index_dir, exist_ok=True)
        
        # Sanitize identifiers for use in filenames
        safe_dataset_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
        safe_embedder_identifier = "".join(c if c.isalnum() else "_" for c in embedder_identifier)
        
        base_filename = f"{safe_dataset_name}_{safe_embedder_identifier}"
        
        self.faiss_index_path = os.path.join(index_dir, f"{base_filename}_faiss.idx")
        self.database_path = os.path.join(index_dir, f"{base_filename}_database.pkl")
        self.bm25_index_path = os.path.join(index_dir, f"{base_filename}_bm25.pkl")

        if not force_rebuild:
            self.load_indices()

        if not self.index:
            if (embedding_source == 'local' and embedder) or \
               (embedding_source == 'openai' and openai_client and self.embedding_model_name):
                print(f"Building FAISS index for {safe_dataset_name} with {safe_embedder_identifier}...")
                if not self.embedding_dim and len(self.train_data_source) > 0 and self.train_data_source[0] and len(self.train_data_source[0]) > 0:
                    try:
                        sample_emb = self._get_embedding(self.train_data_source[0][0])
                        self.embedding_dim = sample_emb.shape[0]
                        print(f"Inferred embedding dimension: {self.embedding_dim}")
                    except Exception as e:
                        print(f"Error: Failed to get sample embedding for {safe_embedder_identifier}: {e}")
                        # return # Decide if this is fatal
                
                if self.embedding_dim:
                    embeddings_list = []
                    for s, _ in tqdm(self.train_data_source, desc=f"Embed ({safe_embedder_identifier})"):
                        embeddings_list.append(self._get_embedding(s))
                    
                    if not embeddings_list:
                        print("Warning: No embeddings generated. FAISS index cannot be built.")
                    else:
                        embs = np.stack(embeddings_list)
                        d = embs.shape[1]
                        if d != self.embedding_dim:
                            print(f"Warning: Actual embedding dimension {d} differs from expected {self.embedding_dim} for {safe_embedder_identifier}. Using actual: {d}.")
                            self.embedding_dim = d
                        self.index = faiss.IndexFlatL2(d)
                        self.index.add(embs)
                        self.train_data_for_faiss = self.train_data_source
                        print(f"FAISS index built with {self.index.ntotal} vectors for {safe_dataset_name}/{safe_embedder_identifier}.")
                else:
                     print(f"FAISS index for {safe_dataset_name}/{safe_embedder_identifier} cannot be built: embedding dimension unknown.")
            else:
                print(f"FAISS index for {safe_dataset_name}/{safe_embedder_identifier} not built: embedder/client not configured.")

        if not self.bm25_index and self.tokenized_corpus_for_bm25:
            print(f"Building BM25 index for {safe_dataset_name}...") # BM25 is embedder-agnostic
            self.bm25_index = BM25Okapi(self.tokenized_corpus_for_bm25)
            print(f"BM25 index built for {len(self.tokenized_corpus_for_bm25)} documents.")
        elif not self.tokenized_corpus_for_bm25 :
             print("BM25 index not built: Tokenized corpus not provided.")


        if (self.index and not os.path.exists(self.faiss_index_path)) or \
           (self.bm25_index and not os.path.exists(self.bm25_index_path)) or \
           force_rebuild:
            self.save_indices()

    def _get_embedding(self, text_to_embed):
        if self.embedding_source == 'local':
            if not self.embedder:
                raise ValueError("Local embedder not provided for _get_embedding call.")
            # Assuming normalize_embeddings is a good default for sentence-transformers
            return self.embedder.encode(text_to_embed, normalize_embeddings=True)
        elif self.embedding_source == 'openai':
            if not self.openai_client or not self.embedding_model_name:
                raise ValueError("OpenAI client or model name not provided for _get_embedding call.")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        input=text_to_embed,
                        model=self.embedding_model_name
                    )
                    embedding = np.array(response.data[0].embedding, dtype=np.float32)
                    norm = np.linalg.norm(embedding)
                    return embedding / norm if norm != 0 else embedding # Normalize
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"OpenAI Embedding Error ({self.embedding_model_name}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"OpenAI Embedding Error ({self.embedding_model_name}): Failed after {max_retries} attempts: {e}")
                        if self.embedding_dim:
                             return np.zeros(self.embedding_dim, dtype=np.float32) # Fallback
                        raise
        else:
            raise ValueError(f"Unsupported embedding source: {self.embedding_source}")

    def save_indices(self) -> None:
        """Save FAISS index, BM25 index, and metadata to disk using configured paths."""
        if self.index and self.train_data_for_faiss: # Ensure data is also present
            try:
                faiss.write_index(self.index, self.faiss_index_path)
                print(f"FAISS index ({self.index.ntotal} vectors) saved to {self.faiss_index_path}.")
                with open(self.database_path, "wb") as f:
                    pickle.dump(self.train_data_for_faiss, f)
                print(f"Database ({len(self.train_data_for_faiss)} entries) saved to {self.database_path}.")
            except Exception as e:
                print(f"Error saving FAISS index or database to {self.faiss_index_path}/{self.database_path}: {e}")
        elif self.index and not self.train_data_for_faiss:
             print(f"Warning: FAISS index exists but no corresponding database (train_data_for_faiss). Cannot save database for {self.faiss_index_path}.")
        else:
            print(f"Warning: FAISS index not initialized. Cannot save to {self.faiss_index_path}.")


        if self.bm25_index and self.tokenized_corpus_for_bm25:
            try:
                bm25_data_to_save = {
                    'bm25_index_obj': self.bm25_index,
                    'tokenized_corpus': self.tokenized_corpus_for_bm25
                }
                with open(self.bm25_index_path, "wb") as f:
                    pickle.dump(bm25_data_to_save, f)
                print(f"BM25 index (for {len(self.tokenized_corpus_for_bm25)} docs) saved to {self.bm25_index_path}.")
            except Exception as e:
                print(f"Error saving BM25 index to {self.bm25_index_path}: {e}")
        else:
             print(f"Warning: BM25 index not initialized or tokenized corpus missing. Cannot save BM25 to {self.bm25_index_path}.")


    def load_indices(self) -> None:
        """Load FAISS index, BM25 index, and metadata if available from configured paths."""
        faiss_loaded_successfully = False

        # Load FAISS and Database
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.database_path):
            print(f"Attempting to load FAISS index ({self.faiss_index_path}) and database ({self.database_path})...")
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                # If embedding_dim is known, validate it.
                if self.embedding_dim and self.index.d != self.embedding_dim:
                    print(f"Error: Loaded FAISS index dim ({self.index.d}) from {self.faiss_index_path} != expected model dim ({self.embedding_dim}). Invalidating.")
                    self.index = None
                else:
                    # If embedding_dim was not known, adopt it from the loaded index.
                    if not self.embedding_dim:
                        self.embedding_dim = self.index.d
                        print(f"Adopted embedding dimension {self.embedding_dim} from loaded FAISS index: {self.faiss_index_path}")
                    
                    with open(self.database_path, "rb") as f:
                        self.train_data_for_faiss = pickle.load(f)
                    print(f"FAISS index loaded ({self.index.ntotal} vectors). Database loaded ({len(self.train_data_for_faiss)} entries).")

                    if self.index.ntotal != len(self.train_data_for_faiss):
                        print(f"Warning: FAISS size ({self.index.ntotal}) != DB size ({len(self.train_data_for_faiss)}) for {self.faiss_index_path}. Check consistency.")
                    # Crucially, train_data_source should align with what train_data_for_faiss represents for retrieval to work.
                    # If they are always set to be the same (train_data_source used to build index), this is fine.
                    faiss_loaded_successfully = True
            except Exception as e:
                print(f"Error loading FAISS index or database from {self.faiss_index_path}/{self.database_path}: {e}. Will try to rebuild.")
                self.index = None
                self.train_data_for_faiss = []
        else:
            if not os.path.exists(self.faiss_index_path):
                 print(f"Saved FAISS index ({self.faiss_index_path}) not found.")
            if not os.path.exists(self.database_path) and os.path.exists(self.faiss_index_path) :
                 print(f"FAISS index ({self.faiss_index_path}) found, but database ({self.database_path}) not found. FAISS index will not be loaded.")
                 self.index = None # Invalidate if database is missing

        # Load BM25 Index
        if os.path.exists(self.bm25_index_path):
            print(f"Attempting to load BM25 index ({self.bm25_index_path})...")
            try:
                with open(self.bm25_index_path, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['bm25_index_obj']
                    # Link the loaded tokenized_corpus if it's part of the saved bm25 data
                    # Or ensure the one passed to __init__ is still relevant/used
                    self.tokenized_corpus_for_bm25 = bm25_data.get('tokenized_corpus', self.tokenized_corpus_for_bm25)

                if not hasattr(self.bm25_index, 'get_scores'):
                    print(f"Error: Loaded BM25 object from {self.bm25_index_path} is not valid. Rebuilding.")
                    self.bm25_index = None
                elif self.tokenized_corpus_for_bm25 and len(self.tokenized_corpus_for_bm25) == 0: # Check if tokenized corpus is empty
                    print(f"Warning: Loaded BM25 index from {self.bm25_index_path}, but tokenized corpus is empty or missing.")
                else:
                    print(f"BM25 index loaded (for approx. {len(self.tokenized_corpus_for_bm25)} documents if available).")

            except Exception as e:
                print(f"Error loading BM25 index from {self.bm25_index_path}: {e}. Will try to rebuild.")
                self.bm25_index = None
        else:
            print(f"Saved BM25 index ({self.bm25_index_path}) not found.")

        if not self.index or (self.tokenized_corpus_for_bm25 and not self.bm25_index):
             print("One or more indices will be (re)built if their components are available and rebuild is not forced off.")

    # ... (random, bm25, dense methods remain largely the same but ensure they use self.train_data_source for actual content)
    # Make sure indices returned by bm25 and dense correctly map to self.train_data_source
    # If train_data_for_faiss IS train_data_source, and tokenized_corpus_for_bm25 indices align with train_data_source, it's fine.

    def random(self, query, k):
        if not self.train_data_source: return []
        num_available = len(self.train_data_source)
        if k > num_available: k = num_available
        if k == 0: return []
        idxs = random.sample(range(num_available), k)
        return [(i, None) for i in idxs] # Indices for self.train_data_source

    def bm25(self, query, k):
        if not self.bm25_index:
            print("Warning: BM25 index not available for retrieval.")
            return []
        if not self.tokenized_corpus_for_bm25 or len(self.tokenized_corpus_for_bm25) == 0 :
            print("Warning: BM25 tokenized corpus is empty or not available.")
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        num_available = len(scores)
        if k > num_available: k = num_available
        if k == 0: return []

        sorted_indices = np.argsort(scores)[::-1]
        results = []
        for i in sorted_indices:
            if len(results) < k and scores[i] > 0:
                 results.append((i, float(scores[i]))) # Indices for self.tokenized_corpus_for_bm25
            elif len(results) == k:
                break
        return results

    def dense(self, query, k):
        if not self.index:
            print("Warning: FAISS index not available for dense retrieval.")
            return []
        if self.index.ntotal == 0:
            print("Warning: FAISS index is empty.")
            return []

        q_emb = self._get_embedding(query)
        if q_emb is None or q_emb.ndim == 0 or q_emb.size == 0 :
            print("Error: Could not generate query embedding for dense search.")
            return []

        num_vectors_in_index = self.index.ntotal
        if k > num_vectors_in_index: k = num_vectors_in_index
        if k == 0: return []
            
        D, I = self.index.search(q_emb.reshape(1, -1).astype(np.float32), k)
        # Indices for self.train_data_for_faiss
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    def hybrid_bm25_dense(self, query, k):
        """
        Retrieve top-k candidates by combining BM25 and dense retrieval scores.
        Scores are normalized (z-score) and summed, then top-k are returned.
        If only one method is available, fallback to it.
        Returns: List of (index, combined_score)
        """
        # Get BM25 scores for all docs
        bm25_results = self.bm25(query, len(self.train_data_source)) if self.bm25_index else []
        dense_results = self.dense(query, len(self.train_data_source)) if self.index else []

        if not bm25_results and not dense_results:
            print("Warning: Neither BM25 nor dense retrieval available for hybrid retrieval.")
            return []
        if not bm25_results:
            print("Warning: BM25 not available, using dense only for hybrid retrieval.")
            return self.dense(query, k)
        if not dense_results:
            print("Warning: Dense not available, using BM25 only for hybrid retrieval.")
            return self.bm25(query, k)

        # Convert to dict for fast lookup
        bm25_dict = dict(bm25_results)
        dense_dict = dict(dense_results)
        all_indices = set(bm25_dict.keys()) | set(dense_dict.keys())

        # Fill missing scores with 0
        bm25_scores = np.array([bm25_dict.get(i, 0.0) for i in all_indices])
        dense_scores = np.array([dense_dict.get(i, 0.0) for i in all_indices])

        # Normalize scores (z-score)
        def zscore(x):
            if np.std(x) == 0:
                return np.zeros_like(x)
            return (x - np.mean(x)) / np.std(x)
        bm25_norm = zscore(bm25_scores)
        dense_norm = zscore(dense_scores)
        combined_scores = bm25_norm + dense_norm

        # Sort by combined score
        sorted_indices = np.array(list(all_indices))[np.argsort(combined_scores)[::-1]]
        sorted_scores = np.sort(combined_scores)[::-1]

        results = [(int(idx), float(score)) for idx, score in zip(sorted_indices, sorted_scores)][:k]
        return results