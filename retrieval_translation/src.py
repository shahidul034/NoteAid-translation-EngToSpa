# src.py
import os
import json
import time
import gc
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer # Assuming this might be needed if embedder is just a name

from retrieval import Retriever # Ensure this is the updated Retriever
from rerank import Reranker
from templates import PromptTemplateManager
from eval import EvaluateMetric

# At the beginning of src.py or within ExperimentRunner class
INITIAL_CANDIDATES_FOR_RERANK = 10 # Number of candidates to fetch for reranking

class ExperimentTracker:
    """Tracks experiment progress and manages checkpoints."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, "experiments")
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        
        # Create necessary directories
        for dir_path in [self.experiments_dir, self.checkpoints_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.current_experiment = None
        self.experiment_start_time = None
    
    def start_experiment(self, config: Dict) -> str:
        """Start a new experiment and return its ID."""
        # Generate experiment ID
        config_str = json.dumps(config, sort_keys=True)
        experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create experiment directory
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save experiment metadata
        metadata = {
            'id': experiment_id,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'progress': {
                'completed_methods': [],
                'current_method': None,
                'current_template': None,
                'current_shots': None,
                'last_batch': -1
            }
        }
        
        metadata_path = os.path.join(self.metadata_dir, f"{experiment_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.current_experiment = experiment_id
        self.experiment_start_time = time.time()
        
        return experiment_id
    
    def update_progress(self, method: str, template: str, shots: int, batch_idx: int):
        """Update experiment progress."""
        if not self.current_experiment:
            return
        
        metadata_path = os.path.join(self.metadata_dir, f"{self.current_experiment}.json")
        if not os.path.exists(metadata_path):
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['progress'].update({
            'current_method': method,
            'current_template': template,
            'current_shots': shots,
            'last_batch': batch_idx
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def complete_method(self, method: str):
        """Mark a method as completed."""
        if not self.current_experiment:
            return
        
        metadata_path = os.path.join(self.metadata_dir, f"{self.current_experiment}.json")
        if not os.path.exists(metadata_path):
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if method not in metadata['progress']['completed_methods']:
            metadata['progress']['completed_methods'].append(method)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_checkpoint_path(self, method: str, template: str, shots: int, batch_idx: int) -> str:
        """Get the path for a specific checkpoint."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        return os.path.join(
            self.checkpoints_dir,
            f"{self.current_experiment}_{method}_{template}_{shots}_batch_{batch_idx}.pkl"
        )
    
    def save_checkpoint(self, method: str, template: str, shots: int, batch_idx: int,
                       raw_out: List[Dict], rerank_out: List[Dict]) -> None:
        """Save checkpoint for a specific batch."""
        checkpoint_path = self.get_checkpoint_path(method, template, shots, batch_idx)
        checkpoint_data = {
            'raw_out': raw_out,
            'rerank_out': rerank_out,
            'batch_idx': batch_idx,
            'timestamp': time.time()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, method: str, template: str, shots: int, batch_idx: int) -> Optional[Dict]:
        """Load checkpoint for a specific batch if it exists."""
        checkpoint_path = self.get_checkpoint_path(method, template, shots, batch_idx)
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
        return None
    
    def get_last_completed_batch(self, method: str, template: str, shots: int) -> int:
        """Find the last completed batch for a specific configuration."""
        if not self.current_experiment:
            return -1
        
        pattern = f"{self.current_experiment}_{method}_{template}_{shots}_batch_*.pkl"
        checkpoints = list(Path(self.checkpoints_dir).glob(pattern))
        if not checkpoints:
            return -1
        return max(int(c.stem.split('_')[-1]) for c in checkpoints)
    
    def finish_experiment(self):
        """Mark the current experiment as completed."""
        if not self.current_experiment:
            return
        
        metadata_path = os.path.join(self.metadata_dir, f"{self.current_experiment}.json")
        if not os.path.exists(metadata_path):
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = 'completed'
        metadata['end_time'] = datetime.now().isoformat()
        metadata['duration'] = time.time() - self.experiment_start_time
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.current_experiment = None
        self.experiment_start_time = None

class ExperimentRunner:
    def __init__(
        self,
        train_data: list,
        test_data: list,
        embedder_obj=None, # Changed from embedder to embedder_obj for clarity
        tokenized_corpus: list = None,
        llm_model_name: str = "gpt-4o-mini", # Renamed from model_name
        embedding_source: str = "local",
        # For local, this is the SBERT model name. For openai, this is the OpenAI model name.
        embedding_model_name_or_path: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        dataset_name: str = "unknown_dataset", # New parameter
        base_dir: str = "experiments"
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)
        self.llm_model_name = llm_model_name # For text generation
        
        self.embedding_source = embedding_source
        # This is the name/path for the embedding model itself, e.g., "text-embedding-3-small" or "S-PubMedBert-MS-MARCO"
        self.embedding_model_name_or_path = embedding_model_name_or_path
        self.dataset_name = dataset_name
        
        # Initialize experiment tracker
        self.tracker = ExperimentTracker(base_dir)
        
        # Generate experiment configuration
        config = {
            'dataset_name': dataset_name,
            'embedding_source': embedding_source,
            'embedding_model': embedding_model_name_or_path,
            'llm_model': llm_model_name
        }
        
        # Start experiment tracking
        self.experiment_id = self.tracker.start_experiment(config)

        # Determine embedder_identifier for file naming
        # Sanitize the model name/path for use in filenames
        _safe_emb_model_name = "".join(c if c.isalnum() else "_" for c in self.embedding_model_name_or_path)
        self.embedder_identifier = f"{self.embedding_source}_{_safe_emb_model_name}"

        # Handle embedder_obj initialization if it's not passed directly
        # This assumes embedder_obj is the actual SentenceTransformer object for 'local'
        if embedding_source == "local" and embedder_obj is None:
            print(f"Initializing local SentenceTransformer: {self.embedding_model_name_or_path}")
            embedder_obj = SentenceTransformer(self.embedding_model_name_or_path)
        elif embedding_source == "openai" and embedder_obj is not None:
            print("Warning: embedder_obj provided for OpenAI source, but it's not used by Retriever's OpenAI path. Ensure this is intended.")

        # Store only necessary data
        self.train_data_source = train_data
        self.test = test_data
        
        # Initialize components
        self.ret = Retriever(
            train_data,
            embedder=embedder_obj, # Pass the actual embedder object for local
            bm25_tokenized=tokenized_corpus,
            embedding_source=self.embedding_source,
            openai_client=self.client if self.embedding_source == "openai" else None,
            embedding_model_name=self.embedding_model_name_or_path if self.embedding_source == "openai" else None, # Pass specific model name for OpenAI
            index_dir=os.path.join("indices_cache"), # Centralized index directory, can be configured
            dataset_name=self.dataset_name,
            embedder_identifier=self.embedder_identifier
        )
        self.rerank = Reranker() # Assuming Reranker doesn't need specific embedder info for init
        self.tplman = PromptTemplateManager()

    def cleanup(self):
        """Clean up resources when done."""
        if hasattr(self, 'ret'):
            del self.ret
        if hasattr(self, 'rerank'):
            del self.rerank
        if hasattr(self, 'tplman'):
            del self.tplman
        if hasattr(self, 'client'):
            del self.client
        gc.collect()
        
        # Mark experiment as completed
        self.tracker.finish_experiment()

    def _generate(self, prompt_text: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in _generate: {str(e)}")
            return ""

    def _process_batch(self, batch: List[Dict], method_name: str, tpl: str, 
                      k_shots: int, initial_retrieval_count: int) -> Tuple[List[Dict], List[Dict]]:
        """Process a batch of test items to reduce memory usage."""
        raw_out, rerank_out = [], []
        
        for item in batch:
            src = item["english"]
            tgt = item["spanish"]
            
            # Process raw examples
            raw_cand_indices_scores = getattr(self.ret, method_name)(src, k_shots)
            raw_prompt_indices = [idx for idx, _ in raw_cand_indices_scores]
            
            raw_prompt_exs = []
            if raw_prompt_indices:
                try:
                    raw_prompt_exs = [self.train_data_source[idx] for idx in raw_prompt_indices]
                except IndexError as e:
                    print(f"Error accessing train_data_source for raw prompt: {e}")
            
            prompt_raw = self.tplman.get(tpl, src, raw_prompt_exs)
            raw_trans = self._generate(prompt_raw)
            raw_out.append({"id": len(raw_out), "source": src, "target": tgt, "prompt": prompt_raw, "trans": raw_trans})
            
            # Process reranked examples
            initial_candidates_scores = getattr(self.ret, method_name)(src, initial_retrieval_count)
            initial_candidate_indices = [idx for idx, _ in initial_candidates_scores]
            
            rr_exs = []
            if initial_candidate_indices:
                try:
                    candidate_texts_for_rerank = [self.train_data_source[idx]['english'] for idx in initial_candidate_indices]
                    reranked_idx_score_pairs = self.rerank.cross_with_indices(src, initial_candidate_indices, candidate_texts_for_rerank)
                    final_reranked_indices = [idx for idx, score in reranked_idx_score_pairs[:k_shots]]
                    if final_reranked_indices:
                        rr_exs = [self.train_data_source[idx] for idx in final_reranked_indices]
                except Exception as e:
                    print(f"Error in reranking process: {str(e)}")
                    rr_exs = raw_prompt_exs[:k_shots]
            else:
                rr_exs = raw_prompt_exs[:k_shots]
            
            if not rr_exs and k_shots > 0:
                rr_exs = raw_prompt_exs[:k_shots]
            
            prompt_rr = self.tplman.get(tpl, src, rr_exs)
            rr_trans = self._generate(prompt_rr)
            rerank_out.append({"id": len(rerank_out), "source": src, "target": tgt, "prompt": prompt_rr, "trans": rr_trans})
        
        return raw_out, rerank_out

    def run(
        self,
        methods: list,
        templates: list,
        shots: list,
        output_dir: str = "results",
        lang: str = "sp",
        batch_size: int = 50
    ):
        current_run_output_dir = os.path.join(output_dir, self.dataset_name, self.embedder_identifier)
        os.makedirs(current_run_output_dir, exist_ok=True)
        
        try:
            for method_name in methods:
                if not hasattr(self.ret, method_name):
                    print(f"Warning: Method '{method_name}' not found in Retriever. Skipping.")
                    continue
                
                # Dictionary to store all results for this method
                method_results = {}
                
                for tpl in templates:
                    for k_shots in shots:
                        start_all = time.time()
                        print(f"Running: Dataset={self.dataset_name}, Embedder={self.embedder_identifier}, Method={method_name}, Template={tpl}, Shots={k_shots}")
                        
                        # Find last completed batch
                        start_batch = 0
                        last_batch = self.tracker.get_last_completed_batch(method_name, tpl, k_shots)
                        if last_batch >= 0:
                            print(f"Resuming from batch {last_batch + 1}")
                            start_batch = last_batch + 1
                        
                        # Process test items in batches
                        all_raw_out, all_rerank_out = [], []
                        
                        # Load previous results if resuming
                        if start_batch > 0:
                            for i in range(start_batch):
                                checkpoint = self.tracker.load_checkpoint(method_name, tpl, k_shots, i)
                                if checkpoint:
                                    all_raw_out.extend(checkpoint['raw_out'])
                                    all_rerank_out.extend(checkpoint['rerank_out'])
                        
                        for i in range(start_batch, len(self.test), batch_size):
                            batch = self.test[i:i + batch_size]
                            batch_idx = i // batch_size
                            
                            # Check if this batch was already processed
                            checkpoint = self.tracker.load_checkpoint(method_name, tpl, k_shots, batch_idx)
                            if checkpoint:
                                print(f"Skipping batch {batch_idx} (already processed)")
                                all_raw_out.extend(checkpoint['raw_out'])
                                all_rerank_out.extend(checkpoint['rerank_out'])
                                continue
                            
                            raw_out, rerank_out = self._process_batch(batch, method_name, tpl, k_shots, INITIAL_CANDIDATES_FOR_RERANK)
                            all_raw_out.extend(raw_out)
                            all_rerank_out.extend(rerank_out)
                            
                            # Save checkpoint for this batch
                            self.tracker.save_checkpoint(method_name, tpl, k_shots, batch_idx, raw_out, rerank_out)
                            
                            # Update progress
                            self.tracker.update_progress(method_name, tpl, k_shots, batch_idx)
                            
                            # Save intermediate results without evaluation
                            if (i + batch_size) % (batch_size * 2) == 0:
                                self._save_results(current_run_output_dir, method_name, tpl, k_shots, all_raw_out, all_rerank_out, evaluate=False)
                                gc.collect()
                        
                        # Store results for this template/shot combination
                        method_results[(tpl, k_shots)] = {
                            'raw_out': all_raw_out,
                            'rerank_out': all_rerank_out
                        }
                        
                        elapsed = time.time() - start_all
                        print(f"-> Finished {self.dataset_name}/{self.embedder_identifier}/{method_name}/{tpl}/{k_shots} in {elapsed:.1f}s")
                
                # After all templates and shots are complete for this method, evaluate and save final results
                for (tpl, k_shots), results in method_results.items():
                    self._save_results(current_run_output_dir, method_name, tpl, k_shots, 
                                     results['raw_out'], results['rerank_out'], evaluate=True)
                
                # Mark method as completed
                self.tracker.complete_method(method_name)
        finally:
            self.cleanup()

    def _save_results(self, output_dir: str, method_name: str, tpl: str, k_shots: int, 
                     raw_out: List[Dict], rerank_out: List[Dict], evaluate: bool = True) -> None:
        """Save results and optionally evaluate metrics."""
        metrics_subdir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_subdir, exist_ok=True)
        
        # Save raw results
        raw_file = os.path.join(output_dir, f"raw_{method_name}_{tpl}_{k_shots}.json")
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_out, f, indent=2, ensure_ascii=False)
        
        # Save reranked results
        rr_file = os.path.join(output_dir, f"rerank_{method_name}_{tpl}_{k_shots}.json")
        with open(rr_file, "w", encoding="utf-8") as f:
            json.dump(rerank_out, f, indent=2, ensure_ascii=False)

        evaluate = False
        
        # Only evaluate metrics if requested
        if evaluate:
            # Evaluate metrics
            ev_raw = EvaluateMetric(datapath=raw_file, output_dir=metrics_subdir, generated_col="trans", reference_col="target", source_col="source")
            res_raw = ev_raw.evaluate_all(langs="sp")
            
            ev_rr = EvaluateMetric(datapath=rr_file, output_dir=metrics_subdir, generated_col="trans", reference_col="target", source_col="source")
            res_rr = ev_rr.evaluate_all(langs="sp")
            
            # Save combined metrics
            combo = {
                "method": method_name,
                "template": tpl,
                "shots": k_shots,
                "dataset": self.dataset_name,
                "embedder": self.embedder_identifier,
                "raw": res_raw,
                "rerank": res_rr
            }
            combo_file = os.path.join(metrics_subdir, f"metrics_summary_{method_name}_{tpl}_{k_shots}.json")
            with open(combo_file, "w", encoding="utf-8") as cf:
                json.dump(combo, cf, indent=2, ensure_ascii=False)