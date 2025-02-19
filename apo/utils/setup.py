import dspy
import json
import os

from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

load_dotenv()

def init_openai_model():
    """
    Initialize and configure the DSPy OpenAI model.
    """
    api_key = os.getenv('OPENAI_API_KEY')  # Load API key from .env

    if not api_key:
        raise ValueError("Missing OpenAI API Key. Please check your .env file.")

    # Initialize OpenAI model for DSPy
    openai_model = dspy.LM('openai/gpt-4o-mini', api_key=api_key)

    # Configure DSPy settings globally
    dspy.settings.configure(lm=openai_model, trace=[])

    return openai_model

def import_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_dataset(json_file):
    """
    Load and process the English-Spanish translation dataset
    
    Args:
        json_file (str): Path to JSON file containing translations
        
    Returns:
        list: List of dspy.Example objects
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data[:50]:
        example = dspy.Example(
            english=item['english'],
            spanish=item['spanish']
        )
        examples.append(example)
    return examples
