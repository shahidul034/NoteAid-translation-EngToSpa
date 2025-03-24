# MEDPROMPT

## Overview

- K-Nearest Neighbors (kNN) context retrieval
- OpenAI embeddings for semantic similarity
- FAISS index for efficient similarity search
- Support for both forward and back translation
- Multiple evaluation metrics including BLEU, ROUGE, BERTScore, and COMET


### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries
- faiss-cpu
- numpy
- openai
- tqdm
- rouge-score
- sacrebleu
- bert-score
- comet-score
- pandas

### API Keys
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Ollama (for local model inference)

## Project Structure

```
medical-translation/
│
├── local.py        # Local Ollama-based translation implementation
├── api.py          # OpenAI API-based translation implementation
├── eval.py         # Evaluation metrics and scoring
├── TrainingDataMinusOverlaps.json  # Training data with no overlaps with test data
└── README.md
```

## Configuration

### Embedding and Translation Settings

Modify the following constants in `local.py` or `api.py`:

- `EMBEDDING_MODEL`: Choose embedding model
- `KNN_K`: Number of nearest neighbor examples to retrieve
- `TRAINING_DATA_PATH`: Path to training data
- `TEST_DATA_PATH`: Path to test data
- `OUTPUT_JSON_PATH`: Output path for translations

## Usage

### Preprocessing Training Data

```python
from local import MedPromptSystem

# Load training data
with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# Initialize and preprocess
medprompt = MedPromptSystem()
medprompt.preprocess(train_data)
```

### Translation

#### Single Text Translation

```python
# English to Spanish
spanish_translation = medprompt.generate_translation("Patient diagnosed with diabetes.")

# Spanish to English (in api.py)
english_translation = medprompt.generate_translation(spanish_text, direction="es_to_en")
```

#### Batch Translation

```python
# Process entire test dataset
medprompt.process_test_set(test_data)
```

## Evaluation

The system supports multiple evaluation metrics:

```python
# After generating translations
evaluator = EvaluateMetric('translated_output.json', 
                            "translated_spanish", 
                            "target_spanish", 
                            "original_english")

# Run specific metrics
evaluator.evaluate("BLEU")
evaluator.evaluate("ROUGE")
evaluator.evaluate("BERTSCORE")
evaluator.evaluate("COMET")
```
