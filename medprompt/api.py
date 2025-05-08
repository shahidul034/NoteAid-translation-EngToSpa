from openai import OpenAI
import faiss
import numpy as np
import pickle
import os
import json
import time
from typing import List, Tuple, Dict, Any
from string import Template as StringTemplate

from eval import EvaluateMetric

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
KNN_K = 3

FAISS_INDEX_PATH = "faiss_index.bin"
DATABASE_PATH = "database.pkl"
TRAINING_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/TrainingDataMinusOverlaps.json"
TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
OUTPUT_JSON_PATH = "translated_output.json"

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class MedPromptSystem:
    def __init__(self):
        self.index = None
        self.train_data = []
        self.load_faiss_index()

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=text, 
                    model=EMBEDDING_MODEL
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  
                    print(f"Error getting embedding: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to get embedding after {max_retries} attempts: {e}")

    def save_faiss_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(DATABASE_PATH, "wb") as f:
                pickle.dump(self.train_data, f)
            print("FAISS index and database saved successfully.")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    def load_faiss_index(self) -> None:
        """Load FAISS index and metadata if available."""
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DATABASE_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                with open(DATABASE_PATH, "rb") as f:
                    self.train_data = pickle.load(f)
                print(f"FAISS index loaded with {self.index.ntotal} vectors.")
                
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Recomputing embeddings...")
                self.index = None  
        else:
            print("No saved FAISS index found. Will create new index during preprocessing.")
            self.index = None

    def preprocess(self, training_data: List[Dict[str, str]]) -> None:
        """Generate embeddings, store in FAISS index, and save."""
        if self.index is not None and self.index.ntotal > 0:
            print(f"FAISS index already exists with {self.index.ntotal} vectors. Skipping preprocessing.")
            return

        print(f"Preprocessing {len(training_data)} training examples...")
        self.train_data = []
        embeddings = []
        
        batch_size = 100
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(training_data)-1)//batch_size + 1}")
            
            for entry in batch:
                english_text = entry["english"]
                try:
                    emb = self.get_embedding(english_text)
                    embeddings.append(emb)
                    self.train_data.append((english_text, entry["spanish"]))
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue

        if not embeddings:
            raise Exception("No embeddings were generated. Check your data and API key.")

        d = len(embeddings[0])
        self.index = faiss.IndexFlatL2(d)
        
        self.index.add(np.array(embeddings))
        print(f"Added {len(embeddings)} vectors to FAISS index.")
        self.save_faiss_index()

    def knn_retrieve(self, query: str) -> List[Tuple[str, str]]:
        """Retrieve kNN examples for a given query."""
        if self.index is None:
            raise Exception("FAISS index not initialized. Run preprocess() first.")

        query_emb = self.get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, KNN_K)
        
        retrieved_examples = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.train_data) and idx >= 0:
                english, spanish = self.train_data[idx]
                retrieved_examples.append((english, spanish, distances[0][i]))
                print(f"Retrieved example with distance {distances[0][i]:.4f}: {english[:50]}...")
            else:
                print(f"Invalid index: {idx}")
                
        return [(ex[0], ex[1]) for ex in retrieved_examples]

    def format_examples_for_prompt(self, examples):
        """Format the kNN examples for the prompt template."""
        formatted_examples = []
        for eng, spa in examples:
            formatted_examples.append({
                "english_text": eng,
                "spanish_translation": spa
            })
        return formatted_examples

    def generate_translation(self, query: str) -> str:
        """Generate translation using kNN context with the medical translation prompt."""
        # Get similar examples using kNN
        knn_examples = self.knn_retrieve(query)
        
        # Format the prompt with retrieved examples and the query
        formatted_examples = self.format_examples_for_prompt(knn_examples)
        
        # Create the prompt with retrieved examples and the current text to translate
        prompt_text = f"""{{% for item in examples %}}## English Medical Text
                    {{{{ item.english_text }}}}

                    ## Spanish Translation
                    {{{{ item.spanish_translation }}}}

                    {{% endfor %}}## English Medical Text
                    {query}
                    ## Spanish Translation
                    """
        
        # Replace the Jinja2 template syntax with actual examples
        for i, example in enumerate(formatted_examples):
            eng_placeholder = "{{ item.english_text }}"
            spa_placeholder = "{{ item.spanish_translation }}"
            
            prompt_text = prompt_text.replace(
                eng_placeholder, 
                example["english_text"], 
                1
            )
            prompt_text = prompt_text.replace(
                spa_placeholder, 
                example["spanish_translation"], 
                1
            )
        
        # Remove the for loop syntax
        prompt_text = prompt_text.replace("{% for item in examples %}", "")
        prompt_text = prompt_text.replace("{% endfor %}", "")

        print(f"Prompt text: {prompt_text}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini as specified
                messages=[
                    {"role": "system", "content": "You are a medical translation expert. Translate the English medical text to Spanish accurately, preserving medical terminology."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating translation: {e}")
            return f"Error: Failed to generate translation."

    def process_test_set(self, test_data: List[Dict[str, str]]) -> None:
        """Process test data and save results in the required JSON format."""
        results = []
        total = len(test_data)
        
        print(f"Processing {total} test examples...")
        for i, entry in enumerate(test_data):
            if i % 10 == 0:
                print(f"Progress: {i}/{total}")
                
            original_english = entry["english"]
            target_spanish = entry["spanish"]
            
            try:
                translated_spanish = self.generate_translation(original_english)
                
                results.append({
                    "original_english": original_english,
                    "target_spanish": target_spanish,
                    "translated_spanish": translated_spanish
                })
            except Exception as e:
                print(f"Error processing test example {i}: {e}")
                continue

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    try:

        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

        print(f"Loaded {len(train_data)} training examples. Loaded {len(test_data)} testing examples.")
        
        medprompt = MedPromptSystem()
        # medprompt.preprocess(train_data)
        medprompt.process_test_set(test_data)

        evaluator = EvaluateMetric('translated_output.json', "translated_spanish", "target_spanish", "original_english")

        # Compute BLEU, ROUGE, BERTScore, and COMET
        evaluator.evaluate("BLEU")
        evaluator.evaluate("ROUGE")
        evaluator.evaluate("BERTSCORE")
        evaluator.evaluate("COMET")
        # evaluator.evaluate("LLM_AS_A_JUDGE")

        # Print results
        print(evaluator.res)
        
    except Exception as e:
        print(f"Error in main execution: {e}")