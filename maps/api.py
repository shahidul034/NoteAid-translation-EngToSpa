import json
import os
import requests
import argparse  # Add for command line arguments
from typing import List, Dict, Any
from string import Template as StringTemplate
from tqdm import tqdm
from eval import EvaluateMetric
from openai import OpenAI  # Add OpenAI client import

# Configuration
TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
OUTPUT_JSON_PATH = "translated_output.json"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_CONFIG = {
    "model": "llama3.1:latest",
    "options": {
        "temperature": 0.3,
        "num_predict": 512,
    }
}

# OpenAI configuration
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
GPT4O_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
}

# MAPS prompt templates
knowledge_mining_prompt = StringTemplate(
    """You are a medical translation expert specializing in English to Spanish translation. 
    
Please analyze the following English medical text and extract the necessary information from it for translation.

1. Keywords: List the essential medical terms or phrases present in the text with their Spanish translations
2. Topic: Identify the specific medical topic(s) being discussed in the text
3. Demonstrations: Provide 2-3 example sentences similar to the text with their Spanish translations. These should be relevant to the abovbe keywords and topics.

English Medical Text to Analyze:
{{ $english_text }}

Format your response as a JSON with the following structure, return nothing other than the pure JSON:
{
  "keywords": [{"english": "term1", "spanish": "translation1"}, ...],
  "topics": ["topic1", "topic2", ...],
  "demonstrations": [{"english": "example1", "spanish": "translation1"}, ...]
}

Analyze only."""
)

translation_integration_prompt = StringTemplate(
    """You are a medical translation expert specializing in English to Spanish translation.

# English Medical Text to Translate:
{{ $english_text }}

# Translation Context
{{ $context }}

Please translate the English medical text only into Spanish, ensuring accuracy of medical terminology and natural fluency. Utilize the provided context if helpful.

Provide your translation in JSON format, return nothing other than the pure JSON::
{
  "response": "Your Spanish translation here"
}"""
)

quality_estimation_prompt = StringTemplate(
    """You are a quality estimation expert for medical translations from English to Spanish.

# Original English Text:
{{ $english_text }}

# Translation Candidates:
{{ $candidates }}

Evaluate each translation candidate based on:
1. Accuracy of medical terminology
2. Preservation of original meaning
3. Natural fluency in Spanish
4. Consistency of terminology

Select the best translation and explain your reasoning.

Provide your response in JSON format, return nothing other than the pure JSON:
{
  "best_candidate_index": <index of best translation (0-based)>,
  "explanation": "<brief explanation of your choice>",
  "selected_translation": "<the selected translation text>"
}"""
)

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
            return "Error: Failed to generate response with Ollama."
            
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

class GPT4oMiniClient:
    """Client for interacting with OpenAI GPT-4o Mini API"""
    
    def __init__(self, client=openai_client):
        self.client = client
        
        # Check if API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key with: export OPENAI_API_KEY='your-key'")
    
    def generate(self, prompt: str, system_prompt: str, 
                model: str = GPT4O_CONFIG["model"],
                temperature: float = GPT4O_CONFIG["temperature"]) -> str:
        """Generate text using GPT-4o Mini."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with GPT-4o Mini: {e}")
            return "Error: Failed to generate response with GPT-4o Mini."

class SimplifiedMAPSTranslator:
    """Simplified MAPS (Multi-Aspect Prompting and Selection) framework for medical translation"""

    def __init__(self, engine_type="ollama", ollama_model=LLM_CONFIG["model"], gpt_model=GPT4O_CONFIG["model"]):
        """
        Initialize MAPS system with selected engine
        Args:
            engine_type: Either "ollama" or "gpt" to choose the engine to use
            ollama_model: Name of the Ollama model to use (if engine_type is "ollama")
            gpt_model: Name of the GPT model to use (if engine_type is "gpt")
        """
        self.engine_type = engine_type.lower()
        
        if self.engine_type == "ollama":
            self.ollama = OllamaClient()
            self.ollama_model = ollama_model
            
            # Verify Ollama model is available
            available_models = self.ollama.list_models()
            if available_models and ollama_model not in available_models:
                print(f"Warning: Model '{ollama_model}' not found in available models: {available_models}")
                print(f"You may need to pull it first with: ollama pull {ollama_model}")
                
        elif self.engine_type == "gpt":
            self.gpt = GPT4oMiniClient()
            self.gpt_model = gpt_model
            print(f"Using GPT model: {gpt_model}")
        else:
            raise ValueError(f"Invalid engine type: {engine_type}. Must be 'ollama' or 'gpt'")

    def knowledge_mining(self, english_text: str) -> Dict:
        """Step 1: Knowledge Mining - Extract keywords, topic, and demonstrations."""
        system_message = (
            "You are a medical and linguistics expert. Analyze the English medical text and extract "
            "keywords, topics, and demonstrations that will help with translation to Spanish. "
            "Be specific about medical terminology."
        )
        
        prompt_text = knowledge_mining_prompt.substitute(english_text=english_text)
        
        try:
            # Use the selected engine
            if self.engine_type == "ollama":
                response = self.ollama.generate(prompt_text, system_message, self.ollama_model)
            else:  # gpt
                response = self.gpt.generate(prompt_text, system_message, self.gpt_model)
                
            # Handle potential different formats between Ollama and GPT
            try:
                knowledge = json.loads(response)
            except json.JSONDecodeError:
                # If GPT doesn't return valid JSON, try to extract from the text
                print("Warning: Response is not valid JSON. Attempting to extract structure...")
                # Simple fallback with empty structure
                knowledge = {"keywords": [], "topics": [], "demonstrations": []}
            
            print("Knowledge mining successful:")
            print(f"- {len(knowledge.get('keywords', []))} keywords extracted")
            print(f"- {len(knowledge.get('topics', []))} topics identified")
            print(f"- {len(knowledge.get('demonstrations', []))} demonstrations provided")
            return knowledge
        except Exception as e:
            print(f"Error in knowledge mining: {e}")
            return {"keywords": [], "topics": [], "demonstrations": []}

    def knowledge_integration(self, english_text: str, knowledge: Dict) -> List[str]:
        """Step 2: Knowledge Integration - Generate translation candidates with knowledge context."""
        # Prepare contexts
        contexts = [
            self._format_full_context(knowledge),
            self._format_keywords_context(knowledge),
            self._format_topic_demos_context(knowledge),
            ""
        ]
        
        system_message = (
            "You are a medical translation expert. Translate the English medical text to Spanish accurately, "
            "preserving medical terminology. Use the provided context if helpful."
        )
        
        translation_candidates = []
        
        # Generate translation candidates using the selected engine
        for ctx in contexts:
            prompt_text = translation_integration_prompt.substitute(
                english_text=english_text,
                context=ctx
            )
            
            try:
                if self.engine_type == "ollama":
                    response = self.ollama.generate(prompt_text, system_message, self.ollama_model)
                else:  # gpt
                    response = self.gpt.generate(prompt_text, system_message, self.gpt_model)
                    
                # Handle potential different formats
                try:
                    translation_json = json.loads(response)
                    translation_text = translation_json.get("response", "").strip()
                except json.JSONDecodeError:
                    # If not valid JSON, use the response as is
                    translation_text = response.strip()
                    
                if translation_text:
                    translation_candidates.append(translation_text)
                    
            except Exception as e:
                print(f"Error generating translation candidate: {e}")
        
        # Filter out empty candidates
        translation_candidates = [c for c in translation_candidates if c]
        print(f"Generated {len(translation_candidates)} translation candidates")
        
        return translation_candidates

    def knowledge_selection(self, english_text: str, candidates: List[str]) -> str:
        """Step 3: Knowledge Selection - Select best translation using quality estimation."""
        if not candidates:
            return "Error: No translation candidates were generated."
        
        if len(candidates) == 1:
            return candidates[0]
        
        system_message = (
            "You are a quality estimation expert for medical translations. "
            "Evaluate the translation candidates and select the best one based on accuracy, "
            "preservation of meaning, fluency, and terminology consistency."
        )
        
        # Format candidates for prompt
        candidates_text = ""
        for i, candidate in enumerate(candidates):
            candidates_text += f"Candidate {i+1}: {candidate}\n\n"
        
        prompt_text = quality_estimation_prompt.substitute(
            english_text=english_text,
            candidates=candidates_text
        )
        
        try:
            # Use the selected engine
            if self.engine_type == "ollama":
                response = self.ollama.generate(prompt_text, system_message, self.ollama_model)
            else:  # gpt
                response = self.gpt.generate(prompt_text, system_message, self.gpt_model)
                
            # Try to parse JSON response
            try:
                selection_json = json.loads(response)
                best_index = selection_json.get("best_candidate_index", 0)
                selected_translation = selection_json.get("selected_translation", "")
                
                # Use the selected translation if provided, otherwise use the candidate
                if selected_translation and selected_translation.strip():
                    return selected_translation
                
                # Ensure index is within bounds
                if best_index < 0 or best_index >= len(candidates):
                    best_index = 0
                    
                return candidates[best_index]
            except json.JSONDecodeError as e:
                # Log the raw response and error details for debugging
                print("JSONDecodeError encountered while parsing response:")
                print(f"Raw response: {response}")
                print(f"Error message: {e}")
                # Default to first candidate if parsing fails
                return candidates[0]
        except Exception as e:
            print(f"Error in knowledge selection: {e}")
            # Default to first candidate if selection fails
            return candidates[0]

    def _format_full_context(self, knowledge: Dict) -> str:
        """Format full context with all knowledge types."""
        keywords_text = ""
        if knowledge.get("keywords"):
            keywords_text = "# Key Medical Terms:\n"
            for kw in knowledge.get("keywords", []):
                keywords_text += f"- {kw.get('english', '')}: {kw.get('spanish', '')}\n"
        
        topics_text = ""
        if knowledge.get("topics"):
            topics_text = "\n# Medical Topic(s):\n"
            for topic in knowledge.get("topics", []):
                topics_text += f"- {topic}\n"
        
        demos_text = ""
        if knowledge.get("demonstrations"):
            demos_text = "\n# Similar Example Translations:\n"
            for i, demo in enumerate(knowledge.get("demonstrations", [])):
                demos_text += f"Example {i+1}:\n"
                demos_text += f"English: {demo.get('english', '')}\n"
                demos_text += f"Spanish: {demo.get('spanish', '')}\n\n"
        
        return keywords_text + topics_text + demos_text

    def _format_keywords_context(self, knowledge: Dict) -> str:
        """Format context with only keywords."""
        keywords_text = "# Key Medical Terms:\n"
        for kw in knowledge.get("keywords", []):
            keywords_text += f"- {kw.get('english', '')}: {kw.get('spanish', '')}\n"
        return keywords_text

    def _format_topic_demos_context(self, knowledge: Dict) -> str:
        """Format context with only topics and demonstrations."""
        topics_text = "# Medical Topic(s):\n"
        for topic in knowledge.get("topics", []):
            topics_text += f"- {topic}\n"
        
        demos_text = "\n# Similar Example Translations:\n"
        for i, demo in enumerate(knowledge.get("demonstrations", [])):
            demos_text += f"Example {i+1}:\n"
            demos_text += f"English: {demo.get('english', '')}\n"
            demos_text += f"Spanish: {demo.get('spanish', '')}\n\n"
        
        return topics_text + demos_text

    def translate(self, english_text: str) -> Dict:
        """Complete MAPS translation process for a single text."""

        # Step 1: Knowledge Mining
        print("\n1. Knowledge Mining...")
        knowledge = self.knowledge_mining(english_text)
        
        # Step 2: Knowledge Integration
        print("\n2. Knowledge Integration...")
        translation_candidates = self.knowledge_integration(english_text, knowledge)
        
        # Step 3: Knowledge Selection
        print("\n3. Knowledge Selection...")
        final_translation = self.knowledge_selection(english_text, translation_candidates)
        
        return {
            "original_english": english_text,
            "knowledge": knowledge,
            "translation_candidates": translation_candidates,
            "final_translation": final_translation
        }

    def process_test_set(self, test_data: List[Dict[str, str]], output_path=OUTPUT_JSON_PATH) -> None:
        """Process test data and save results in the required JSON format."""
        results = []
        total = len(test_data)
        
        print(f"Processing {total} test examples with MAPS framework using {self.engine_type.upper()}...")
        for i, entry in enumerate(tqdm(test_data, desc="Processing test examples")):
            original_english = entry["english"]
            target_spanish = entry["spanish"]
            
            try:
                # Run complete MAPS process
                translation_result = self.translate(original_english)
                
                # Format result for output
                results.append({
                    "original_english": original_english,
                    "target_spanish": target_spanish,
                    "translated_spanish": translation_result["final_translation"],
                    "knowledge_mining": translation_result["knowledge"],
                    "translation_candidates": translation_result["translation_candidates"]
                })
                
                print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                
            except Exception as e:
                print(f"Error processing test example {i}: {e}")
                continue

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    try:

        engine = 'gpt'

        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"Loaded {len(test_data)} testing examples.")

        # Create MAPS translator with selected engine
        maps_translator = SimplifiedMAPSTranslator(
            engine_type="gpt",
            gpt_model="gpt-4o-mini"
        )

        # sample_text = "Bili lights is a type of light therapy (phototherapy) that is used to treat newborn jaundice . Jaundice is a yellow coloring of the skin and eyes caused by too much of a substance called bilirubin."
        # # Run translation on a sample text
        # translation_result = maps_translator.translate(sample_text)
        # print(f"\n--- Translation Result ---")
        # print(translation_result)

        # Process test set with MAPS framework
        maps_translator.process_test_set(test_data, "translated_output.json")
        
        # Run evaluation
        evaluator = EvaluateMetric("translated_output.json", "translated_spanish", "target_spanish", "original_english")
        evaluator.evaluate("BLEU")
        evaluator.evaluate("ROUGE")
        evaluator.evaluate("BERTSCORE")
        evaluator.evaluate("COMET")
        print(f"\n--- Evaluation Results (GPT) ---")
        print(evaluator.res)
        
    except Exception as e:
        print(f"Error in main execution: {e}")