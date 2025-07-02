import os
import json
import time
from openai import OpenAI
from typing import List, Dict

# Configuration
TEST_DATA_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/scielo/es_en_test_aligned.jsonl"
OUTPUT_JSON_PATH = "direct_translated_output.json"

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class DirectTranslationSystem:
    def __init__(self):
        self.model = "gpt-4o-mini" 
        
    def generate_translation(self, text: str, direction: str = "en_to_es") -> str:
        """
        Generate translation using a direct translation approach with a medical-focused system prompt.
        
        Args:
            text (str): Text to translate
            direction (str): Translation direction - 'en_to_es' or 'es_to_en'
        """
        # System prompt varies based on translation direction
        if direction == "en_to_es":
            system_prompt = "You are a medical translation expert. Translate the English medical text to Spanish accurately, preserving medical terminology and technical biomedical language. Maintain the same level of formality and technical precision as the source text."
            prompt_text = f"Translate the following English medical text to Spanish:\n\n{text}"
        else:  # es_to_en
            system_prompt = "You are a medical translation expert. Translate the Spanish medical text to English accurately, preserving medical terminology and technical biomedical language. Maintain the same level of formality and technical precision as the source text."
            prompt_text = f"Translate the following Spanish medical text to English:\n\n{text}"

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3  # Lower temperature for more consistent translations
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating translation: {e}")
            return f"Error: Failed to generate translation."

    def process_test_set(self, test_data: List[Dict[str, str]]) -> None:
        """
        Process test data with translation and save results.
        
        Args:
            test_data (List[Dict[str, str]]): Test dataset containing English and Spanish texts
        """
        results = []
        total = len(test_data)
        
        print(f"Processing {total} test examples...")
        for i, entry in enumerate(test_data):
            if i % 10 == 0:
                print(f"Progress: {i}/{total}")
                
            original_english = entry["english"]
            target_spanish = entry["spanish"]
            
            try:
                # Direct Translation: English to Spanish
                translated_spanish = self.generate_translation(original_english, "en_to_es")
                
                results.append({
                    "original_english": original_english,
                    "target_spanish": target_spanish,
                    "translated_spanish": translated_spanish
                })
                
                # Add delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing test example {i}: {e}")
                time.sleep(5)  # Longer delay on error
                continue

        # Save translation results
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Translation results saved to {OUTPUT_JSON_PATH}")

def evaluate_results(output_path):
    """
    Evaluate translation results using various metrics.
    This is a placeholder - you would need to implement or import evaluation functions.
    """
    try:
        from eval import EvaluateMetric
        
        evaluator = EvaluateMetric(output_path, "translated_spanish", "target_spanish", "original_english")
        
        print("Translation Metrics:")
        evaluator.evaluate("BLEU")
        evaluator.evaluate("ROUGE")
        evaluator.evaluate("BERTSCORE")
        evaluator.evaluate("COMET")
        evaluator.evaluate("CHRF")
    except ImportError:
        print("Evaluation module not available. Install required packages for evaluation.")

if __name__ == "__main__":
    try:
        # Load test data
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f]
        
        print(f"Loaded {len(test_data)} testing examples.")
        4
        # Create translation system and process test data
        translator = DirectTranslationSystem()
        translator.process_test_set(test_data)
        
        # Evaluate results
        evaluate_results(OUTPUT_JSON_PATH)
        
    except Exception as e:
        print(f"Error in main execution: {e}")