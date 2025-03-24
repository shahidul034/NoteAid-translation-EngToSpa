import os
import json
import time
import faiss
import numpy as np
from openai import OpenAI
import pickle
from typing import List, Tuple, Dict, Any

from eval import EvaluateMetric

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
KNN_K = 3

FAISS_INDEX_PATH = "faiss_index.bin"
DATABASE_PATH = "database.pkl"
TRAINING_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/TrainingDataMinusOverlaps.json"
TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
OUTPUT_JSON_PATH = "translated_output.json"
BACK_TRANSLATION_OUTPUT_PATH = "back_translated_output.json"

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

    def generate_translation(self, query: str, direction: str = "en_to_es") -> str:
        """
        Generate translation using kNN context with the medical translation prompt.
        
        Args:
            query (str): Text to translate
            direction (str): Translation direction - 'en_to_es' or 'es_to_en'
        """
        # Get similar examples using kNN
        knn_examples = self.knn_retrieve(query)
        
        # Format the prompt with retrieved examples and the query
        formatted_examples = self.format_examples_for_prompt(knn_examples)
        
        prompt_text = "Here are some similar medical texts in English and their Spanish translations:\n\n"

        for example in formatted_examples:
            # Create the prompt with retrieved examples and the current text to translate
            prompt_text += "## English Medical Text\n"
            prompt_text += f"{example['english_text']}\n\n"
            prompt_text += "## Spanish Translation\n"
            prompt_text += f"{example['spanish_translation']}\n\n"

        # System prompt varies based on translation direction
        if direction == "en_to_es":
            system_prompt = "You are a medical translation expert. Translate the English medical text to Spanish accurately, preserving medical terminology."
            prompt_text += "## Your task is to translate the following English medical text to Spanish accurately, using the examples above as context:\n"
            prompt_text += f"## English Medical Text\n{query}\n## Spanish Translation\n"
        else:  # es_to_en
            system_prompt = "You are a medical translation expert. Translate the Spanish medical text to English accurately, preserving medical terminology."
            prompt_text += "## Your task is to translate the following Spanish medical text to English accurately, using the examples above as context:\n"
            prompt_text += f"## Spanish Medical Text\n{query}\n## English Translation\n"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating translation: {e}")
            return f"Error: Failed to generate translation."

    def process_test_set(self, test_data: List[Dict[str, str]]) -> None:
        """
        Process test data with forward and back translation, and save results.
        
        Args:
            test_data (List[Dict[str, str]]): Test dataset containing English and Spanish texts
        """
        forward_results = []
        back_translation_results = []
        total = len(test_data)
        
        print(f"Processing {total} test examples...")
        for i, entry in enumerate(test_data):
            if i % 10 == 0:
                print(f"Progress: {i}/{total}")
                
            original_english = entry["english"]
            target_spanish = entry["spanish"]
            
            try:
                # Forward Translation: English to Spanish
                translated_spanish = self.generate_translation(original_english, "en_to_es")
                
                # Back Translation: Spanish to English
                back_translated_english = self.generate_translation(translated_spanish, "es_to_en")
                
                forward_results.append({
                    "original_english": original_english,
                    "target_spanish": target_spanish,
                    "translated_spanish": translated_spanish
                })
                
                back_translation_results.append({
                    "original_english": original_english,
                    "translated_spanish": translated_spanish,
                    "back_translated_english": back_translated_english
                })
                
            except Exception as e:
                print(f"Error processing test example {i}: {e}")
                continue

        # Save forward translation results
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(forward_results, f, indent=4, ensure_ascii=False)
        print(f"Forward translation results saved to {OUTPUT_JSON_PATH}")
        
        # Save back translation results
        with open(BACK_TRANSLATION_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(back_translation_results, f, indent=4, ensure_ascii=False)
        print(f"Back translation results saved to {BACK_TRANSLATION_OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        # with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
        #     train_data = json.load(f)
            
        # with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        #     test_data = json.load(f)
        
        # print(f"Loaded {len(train_data)} training examples. Loaded {len(test_data)} testing examples.")
        
        # medprompt = MedPromptSystem()
        # medprompt.preprocess(train_data)
        # medprompt.process_test_set(test_data)

        # Evaluate both forward and back translation
        # forward_evaluator = EvaluateMetric(OUTPUT_JSON_PATH, "translated_spanish", "target_spanish", "original_english")
        # back_translation_evaluator = EvaluateMetric(BACK_TRANSLATION_OUTPUT_PATH, "back_translated_english", "original_english", "original_english")

        forward_evaluator = EvaluateMetric("/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/results/gpt4o_mini/translated_output.json", "translated_spanish", "target_spanish", "original_english")
        back_translation_evaluator = EvaluateMetric("/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/results/gpt4o_mini/back_translated_output.json", "back_translated_english", "original_english", "original_english")

        # Compute metrics for forward and back translation
        print("Forward Translation Metrics:")
        # forward_evaluator.evaluate("BLEU")
        # forward_evaluator.evaluate("ROUGE")
        # forward_evaluator.evaluate("BERTSCORE")
        # forward_evaluator.evaluate("COMET")
        forward_evaluator.evaluate("CHRF")

        print("\nBack Translation Metrics:")
        # back_translation_evaluator.evaluate("BLEU")
        # back_translation_evaluator.evaluate("ROUGE")
        # back_translation_evaluator.evaluate("BERTSCORE", lang="en")
        # back_translation_evaluator.evaluate("COMET")
        back_translation_evaluator.evaluate("CHRF")
        
    except Exception as e:
        print(f"Error in main execution: {e}")