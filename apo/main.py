import logging
import json
import random
from sklearn.model_selection import train_test_split
from utils.setup import init_openai_model

class TranslationPipeline:
    """Handles Training, Testing, and Evaluation of Translation Models"""

    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.translator = None

    def initialize(self, mode):
        """Initialize translation model and load OpenAI model"""
        try:
            from src.TranslationSystem import TranslationSystem
            import dspy

            self.translator = TranslationSystem()
            openai_model = init_openai_model()
            dspy.configure(lm=openai_model)

            # If test/eval, load the saved model
            if mode in ['test', 'eval']:
                self.translator.compiled_model = dspy.load(self.model_path)
                logging.info("Model successfully initialized and loaded.")
            else:
                logging.info("Model successfully initialized.")
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def load_data(self, split=True):
        """Load dataset and optionally split into train/test sets"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if split:
                train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
                logging.info(f"Loaded dataset with {len(train_set)} training and {len(test_set)} test samples.")
                return train_set, test_set
            else:
                logging.info(f"Loaded dataset with {len(data)} samples.")
                return data, None
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def train_model(self, train_set):
        """Train the translation model on the training set"""
        try:
            self.translator.compile_model(
                dataset_path=self.dataset_path,
                model_save_path=self.model_path
            )
            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self, test_set):
        """Evaluate the model on a test set"""
        try:
            evaluation_results = self.translator.evaluate_test_set(test_set)
            logging.info(f"Test Set Evaluation:\n{evaluation_results}")
            return evaluation_results
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

    def run_example_translation(self, text, reference):
        """Translate a sample text and compute metrics"""
        try:
            result = self.translator.translate_with_metrics(text, reference, return_metrics=True)

            print(f"English: {text}")
            print(f"Spanish: {result['translation']}")
            if result['metrics']:
                print(f"Metrics: {result['metrics']}")

            return result
        except Exception as e:
            logging.error(f"Error during example translation: {str(e)}")
            raise

if __name__ == "__main__":
    try:

        # /Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/Sampled_1000_MedlinePlus_eng_spanish_pair.json
        # /Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json
        # Define paths
        MODEL_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/apo/models/heavy_run2"
        DATASET_PATH = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/all_tran_data/testing data/Sampled_100_MedlinePlus_eng_spanish_pair.json"

        # Initialize pipeline
        pipeline = TranslationPipeline(MODEL_PATH, DATASET_PATH)

        mode = 'eval'  # Change to 'train' for training mode

        pipeline.initialize(mode=mode)

        if mode == 'train':
            # Load and split dataset
            train_data, test_data = pipeline.load_data(split=True)

            # Train the model
            pipeline.train_model(train_data)

            # Evaluate the model
            evaluation_results = pipeline.evaluate_model(test_data)
            print("Final Evaluation:", evaluation_results)
        else:
            # Load dataset without splitting
            test_data, _ = pipeline.load_data(split=False)

            # Evaluate the model
            evaluation_results = pipeline.evaluate_model(test_data)
            print("Final Evaluation:", evaluation_results)

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise
