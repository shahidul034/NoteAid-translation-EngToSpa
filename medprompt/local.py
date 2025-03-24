import faiss
import numpy as np
import pickle
import os
import json
import time
import requests
from typing import List, Tuple, Dict, Any, Optional
from string import Template as StringTemplate

from eval import EvaluateMetric
from openai import OpenAI
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small" 
KNN_K = 3

FAISS_INDEX_PATH = "faiss_index.bin"
DATABASE_PATH = "database.pkl"
TRAINING_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/TrainingDataMinusOverlaps.json"
TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
OUTPUT_JSON_PATH = "translated_output.json"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_CONFIG = {
    # "model": "llama3.1:latest",  
    "model": "phi4",
    "options": {
        "temperature": 0.3,
        "num_predict": 512,
    }
}

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Medical translation prompt definition
medical_translation_prompt = {
    "prompt_name": "medical_translation_english_to_spanish",
    "response_type": "translation",
    "prompt": StringTemplate(
        """{% for item in examples %}## English Medical Text
{{ item.english_text }}

## Spanish Translation
{{ item.spanish_translation }}

{% endfor %}## English Medical Text
{{ english_text }}
## Spanish Translation
"""
    ),
    "examples": [
        {
            "english_text": """The patient presents with acute myocardial infarction with ST-segment elevation in the anterior leads. Start dual antiplatelet therapy with aspirin 325mg and clopidogrel 600mg loading dose, followed by 75mg daily. Administer IV heparin according to weight-based protocol.""",
            "spanish_translation": """El paciente presenta infarto agudo de miocardio con elevación del segmento ST en las derivaciones anteriores. Iniciar terapia antiagregante plaquetaria dual con aspirina 325mg y clopidogrel 600mg como dosis de carga, seguido de 75mg diarios. Administrar heparina IV según protocolo basado en el peso.""",
        },
        {
            "english_text": """Patient diagnosed with type 2 diabetes mellitus. Current HbA1c is 8.2%. Recommend lifestyle modifications including diet and exercise. Start metformin 500mg twice daily with meals, titrate up as tolerated. Monitor blood glucose levels and return for follow-up in 3 months.""",
            "spanish_translation": """Paciente diagnosticado con diabetes mellitus tipo 2. HbA1c actual es 8.2%. Se recomiendan modificaciones en el estilo de vida, incluyendo dieta y ejercicio. Iniciar metformina 500mg dos veces al día con las comidas, aumentar gradualmente según tolerancia. Monitorear niveles de glucosa en sangre y regresar para seguimiento en 3 meses.""",
        },
        {
            "english_text": """The patient is a 68-year-old female with chronic obstructive pulmonary disease (COPD) presenting with increased dyspnea, productive cough with yellowish sputum, and low-grade fever for the past 3 days. Initiate treatment with bronchodilators, antibiotics, and consider a short course of oral corticosteroids.""",
            "spanish_translation": """La paciente es una mujer de 68 años con enfermedad pulmonar obstructiva crónica (EPOC) que presenta aumento de disnea, tos productiva con esputo amarillento y fiebre de bajo grado durante los últimos 3 días. Iniciar tratamiento con broncodilatadores, antibióticos y considerar un curso corto de corticosteroides orales.""",
        },
        {
            "english_text": """Patient reports severe headache, photophobia, and neck stiffness. Physical examination reveals positive Kernig's and Brudzinski's signs. Suspect bacterial meningitis. Order immediate lumbar puncture, blood cultures, and start empiric antibiotic therapy with ceftriaxone and vancomycin.""",
            "spanish_translation": """Paciente reporta cefalea severa, fotofobia y rigidez de nuca. El examen físico revela signos positivos de Kernig y Brudzinski. Se sospecha meningitis bacteriana. Ordenar punción lumbar inmediata, hemocultivos e iniciar terapia antibiótica empírica con ceftriaxona y vancomicina.""",
        },
        {
            "english_text": """The patient presents with right lower quadrant abdominal pain, nausea, vomiting, and low-grade fever. Laboratory results show leukocytosis. Clinical diagnosis of acute appendicitis. Schedule for emergency appendectomy after obtaining informed consent. Administer preoperative antibiotics.""",
            "spanish_translation": """El paciente presenta dolor abdominal en el cuadrante inferior derecho, náuseas, vómitos y fiebre de bajo grado. Los resultados de laboratorio muestran leucocitosis. Diagnóstico clínico de apendicitis aguda. Programar para apendicectomía de emergencia después de obtener consentimiento informado. Administrar antibióticos preoperatorios.""",
        },
    ]
}

class OllamaClient:
    """Client for interacting with Ollama API"""
    def __init__(self, base_url=OLLAMA_BASE_URL):
        self.base_url = base_url
        self.generate_endpoint = f"{base_url}/api/generate"
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{base_url}/api/version")
            if response.status_code == 200:
                print(f"Connected to Ollama server: {response.json().get('version', 'unknown version')}")
            else:
                print(f"Ollama server responded with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Error connecting to Ollama server: {e}")
            print("Make sure Ollama is running on your system")
            raise

    def generate(self, prompt: str, system_prompt: str, 
             model: str = LLM_CONFIG["model"], 
             options: Dict = None) -> str:
        """Generate text using Ollama."""
        if options is None:
            options = LLM_CONFIG["options"]

        payload = {
            "model": model,
            "prompt": prompt,
            "options": options,
            "system": system_prompt,
            "format": "json",
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(self.generate_endpoint, json=payload, headers=headers, stream=True)
            response.raise_for_status()  
            
            generated_text = ""
            for chunk in response.iter_lines():
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    generated_text += data.get("response", "")

            return generated_text.strip()
        
        except requests.exceptions.RequestException as e:
            print(f"Error generating with Ollama: {e}")
            return "Error: Failed to generate translation with Ollama."

            
    def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])]
                return models
            else:
                print(f"Error: Ollama API returned status {response.status_code}")
                return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

class MedPromptSystem:

    def __init__(self, model_name: str = LLM_CONFIG["model"]):
        """
        Initialize MedPrompt system with Ollama
        Args:
            model_name: Name of the Ollama model to use
        """
        self.index = None
        self.train_data = []
        self.ollama = OllamaClient()
        self.model_name = model_name
        
        # Verify model is available
        available_models = self.ollama.list_models()
        if available_models and model_name not in available_models:
            print(f"Warning: Model '{model_name}' not found in available models: {available_models}")
            print(f"You may need to pull it first with: ollama pull {model_name}")
        
        # Load FAISS index
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
            raise Exception("No embeddings were generated. Check your data and Ollama setup.")

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

        # Construct system message
        system_message = (
            "You are a medical translation expert. Translate the English medical text to Spanish accurately, "
            "preserving medical terminology. Use the given examples as reference. Respond using JSON with the following format:\n"
            "{ \"response\": \"Your translated text here\" }"
        )

        # Construct prompt text with kNN few-shot examples
        example_texts = "\n\n".join(
            f"## English Medical Text:\n{ex['english_text']}\n\n"
            f"## Spanish Translation:\n{ex['spanish_translation']}"
            for ex in formatted_examples
        )

        prompt_text = f"{example_texts}\n\n## English Medical Text:\n{query}\n\n## Spanish Translation:"

        # print(f"Final Prompt:\n{prompt_text}")

        try:
            return self.ollama.generate(prompt_text, system_message, self.model_name)
        except Exception as e:
            print(f"Error generating translation: {e}")
            return "Error: Failed to generate translation."

    def process_test_set(self, test_data: List[Dict[str, str]]) -> None:
        """Process test data and save results in the required JSON format."""
        results = []
        total = len(test_data)
        
        print(f"Processing {total} test examples...")
        for i, entry in enumerate(tqdm(test_data, desc="Processing test examples")):
            original_english = entry["english"]
            target_spanish = entry["spanish"]
            
            try:
                translated_response = self.generate_translation(original_english)
                translated_json = json.loads(translated_response)
                translated_spanish = translated_json.get("response", "").strip()
                
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

    # Run ollama pull "insert model here" to pull the model before running the code, also update the same in the code

    try:
        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"Loaded {len(train_data)} training examples. Loaded {len(test_data)} testing examples.")

        
        # Create MedPrompt system using Ollama
        medprompt = MedPromptSystem(model_name="phi4")
        
        # Check which models are available in Ollama
        available_models = medprompt.ollama.list_models()
        print(f"Available Ollama models: {available_models}")
        
        # Run preprocessing if needed
        medprompt.preprocess(train_data)

        # Example usage for running model on a single text
        # example_english = "The patient was diagnosed with hypertension and prescribed lisinopril 10mg daily."
        # spanish_translation = medprompt.generate_translation(example_english)
        # translated_json = json.loads(spanish_translation)
        # translated_spanish = translated_json.get("response", "").strip()
        # print(f"Translation: {translated_spanish}")
        
        # Process test set
        medprompt.process_test_set(test_data)
        
        # # Run evaluation if needed
        evaluator = EvaluateMetric('translated_output.json', "translated_spanish", "target_spanish", "original_english")
        evaluator.evaluate("BLEU")
        evaluator.evaluate("ROUGE")
        evaluator.evaluate("BERTSCORE")
        evaluator.evaluate("COMET")
        print(evaluator.res)
        
    except Exception as e:
        print(f"Error in main execution: {e}")