# main.py
from src import ExperimentRunner
from sentence_transformers import SentenceTransformer
import json
import argparse
import os
from dotenv import load_dotenv
from contextlib import contextmanager
import gc
import logging
from pathlib import Path
from typing import List, Dict
import itertools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run translation experiments with retrieval')
    parser.add_argument('--dataset_name', choices=['sentence', 'scielo'], default='sentence',
                        help='Name of the dataset to use: sentence or scielo')
    parser.add_argument('--embedding_source', choices=['local', 'openai'], default='openai',
                        help='Source for embeddings: local (sentence-transformers) or openai')
    parser.add_argument('--embedding_model_name_or_path', default='text-embedding-3-small',
                        help='Model name for OpenAI embeddings or path/name for local sentence-transformer')
    parser.add_argument('--llm_model_name', default='gpt-4o-mini', help='LLM model for generation')
    parser.add_argument('--methods', nargs='+', default=['random', 'bm25', 'dense', 'hybrid_bm25_dense'],
                        help='Retrieval methods to use')
    parser.add_argument('--templates', nargs='+', default=['T1', 'T2', 'T3', 'T4', 'T5'],
                        help='Prompt templates to use')
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 3, 5],
                        help='Number of shots (examples) to use')
    parser.add_argument('--output_dir', default='experiment_results',
                        help='Base directory to store experiment results')
    parser.add_argument('--force_rebuild_indices', action='store_true',
                        help='Force rebuild of all indices')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of items to process in each batch')
    parser.add_argument('--run_all', action='store_true',
                        help='Run experiments for all combinations of datasets and embedding models')
    parser.add_argument('--embedding_models', nargs='+', default=[
        'pritamdeka/S-PubMedBert-MS-MARCO'
    ], help='List of embedding models to use when running all experiments')
    return parser.parse_args()

@contextmanager
def safe_file_open(filepath, mode='r', encoding='utf-8'):
    """Context manager for safely opening and closing files."""
    try:
        with open(filepath, mode, encoding=encoding) as f:
            yield f
    except Exception as e:
        logging.error(f"Error opening file {filepath}: {str(e)}")
        raise

def load_jsonl_generator(filepath):
    """Generator to load JSONL files line by line."""
    with safe_file_open(filepath) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {filepath}: {str(e)}")
                continue

def load_dataset(dataset_name: str) -> tuple:
    """Load dataset based on name."""
    if dataset_name == 'scielo':
        train_path = "data/es_en_train_aligned.jsonl"
        test_path = "data/es_en_test_aligned.jsonl"
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            raise FileNotFoundError(f"Scielo data files not found at {train_path} or {test_path}")
        train = list(load_jsonl_generator(train_path))
        test = list(load_jsonl_generator(test_path))
    else:  # default to sentence dataset
        train_path = 'data/TrainingDataMinusOverlaps.json'
        test_path = 'data/Sampled_100_MedlinePlus_eng_spanish_pair.json'
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            raise FileNotFoundError(f"Sentence data files not found at {train_path} or {test_path}")
        with safe_file_open(train_path) as f:
            train = json.load(f)
        with safe_file_open(test_path) as f:
            test = json.load(f)
    return train, test

def prepare_bm25_corpus(train_data: List[Dict]) -> List[List[str]]:
    """Prepare tokenized corpus for BM25."""
    if not train_data:
        return None
    
    if isinstance(train_data[0], dict) and "english" in train_data[0]:
        return [x["english"].lower().split() for x in train_data]
    elif isinstance(train_data[0], (list, tuple)) and len(train_data[0]) > 0:
        return [x[0].lower().split() for x in train_data]
    else:
        logging.warning("Could not determine structure of train_data for BM25 tokenization. BM25 might not work.")
        return None

def run_experiment(
    dataset_name: str,
    embedding_source: str,
    embedding_model: str,
    llm_model: str,
    methods: List[str],
    templates: List[str],
    shots: List[int],
    output_dir: str,
    batch_size: int,
    force_rebuild: bool
) -> None:
    """Run a single experiment configuration."""
    try:
        # Load dataset
        train, test = load_dataset(dataset_name)
        
        # Prepare BM25 corpus if needed
        tokenized_corpus = None
        if 'bm25' in methods or 'hybrid_bm25_dense' in methods:
            tokenized_corpus = prepare_bm25_corpus(train)
        
        # Initialize embedder if using local source
        embedder_obj = None
        if embedding_source == 'local':
            embedder_obj = SentenceTransformer(embedding_model)
        
        # Create experiment runner
        runner = ExperimentRunner(
            train_data=train,
            test_data=test,
            embedder_obj=embedder_obj,
            tokenized_corpus=tokenized_corpus,
            llm_model_name=llm_model,
            embedding_source=embedding_source,
            embedding_model_name_or_path=embedding_model,
            dataset_name=dataset_name
        )
        
        if hasattr(runner, 'ret') and runner.ret is not None:
            runner.ret.force_rebuild = force_rebuild
        
        # Run experiment
        runner.run(
            methods=methods,
            templates=templates,
            shots=shots,
            output_dir=output_dir,
            batch_size=batch_size
        )
        
    except Exception as e:
        logging.error(f"Error running experiment for {dataset_name} with {embedding_model}: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        if 'runner' in locals():
            del runner
        if 'train' in locals():
            del train
        if 'test' in locals():
            del test
        if 'tokenized_corpus' in locals():
            del tokenized_corpus
        if 'embedder_obj' in locals():
            del embedder_obj
        gc.collect()

def main():
    args = parse_arguments()
    load_dotenv()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.run_all:
        # Run experiments for all combinations
        datasets = ['sentence', 'scielo']
        embedding_sources = ['local', 'openai']
        
        for dataset, source, model in itertools.product(datasets, embedding_sources, args.embedding_models):
            logging.info(f"Starting experiment: Dataset={dataset}, Source={source}, Model={model}")
            run_experiment(
                dataset_name=dataset,
                embedding_source=source,
                embedding_model=model,
                llm_model=args.llm_model_name,
                methods=args.methods,
                templates=args.templates,
                shots=args.shots,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                force_rebuild=args.force_rebuild_indices
            )
    else:
        # Run single experiment
        run_experiment(
            dataset_name=args.dataset_name,
            embedding_source=args.embedding_source,
            embedding_model=args.embedding_model_name_or_path,
            llm_model=args.llm_model_name,
            methods=args.methods,
            templates=args.templates,
            shots=args.shots,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            force_rebuild=args.force_rebuild_indices
        )

if __name__ == "__main__":
    main()