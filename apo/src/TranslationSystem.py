import dspy
import json
import numpy as np
from typing import List, Dict, Union, Optional
from dspy import Example
from dspy.teleprompt import MIPROv2
from tqdm import tqdm
import logging
from datetime import datetime
import os
from utils.setup import init_openai_model
from modules.TranslationModule import TranslationModule
from src.TranslationEvaluator import TranslationEvaluator
from src.TranslationMetrics import TranslationMetrics

class TranslationSystem:
    """Enhanced translation system with comprehensive evaluation and optimization logging"""
    
    def __init__(self, model_config: Dict = None):
        self.model_config = model_config or {'auto': 'light'}
        self.evaluator = TranslationEvaluator()
        self.compiled_model = None
        
    def compile_model(self, 
                     dataset_path: str,
                     model_save_path: str = None) -> None:
        """
        Compile and optimize the translation model with logging
        
        Args:
            dataset_path: Path to training dataset
            model_save_path: Optional path to save the compiled model
        """
        try:
            # Initialize OpenAI model
            openai_model = init_openai_model()
            
            # Load and validate dataset
            translation_examples = self.load_and_validate_dataset(dataset_path)
            
            teleprompter = MIPROv2(
                prompt_model=openai_model,
                task_model=openai_model,
                metric=self._quality_check_wrapper,
                track_stats=True, 
                **self.model_config
            )
                        
            # Compile model
            self.compiled_model = teleprompter.compile(
                TranslationModule(),
                trainset=translation_examples,
                minibatch=True,
            )
                    
            # Save if path provided
            if model_save_path:
                self.save_model(model_save_path)
                
        except Exception as e:
            logging.error(f"Error during model compilation: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save the compiled model"""
        if self.compiled_model:
            try:
                # Save the model
                self.compiled_model.save(path, save_program=True)
                self.compiled_model.save('/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/apo/test.json', save_program=False)
                
            except Exception as e:
                logging.error(f"Error saving model and logs: {str(e)}")
                raise
        else:
            raise ValueError("No compiled model to save")
            
    def evaluate_test_set(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate model performance on a test set
        
        Args:
            test_data: List of translation pairs
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.compiled_model:
            raise ValueError("Model not compiled. Call compile_model first.")
            
        all_metrics = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            try:
                translation = self.translate_with_metrics(
                    item['english'],
                    target=item['spanish']
                )
                metrics = self.evaluator.calculate_metrics(
                    item['spanish'],
                    translation
                )
                all_metrics.append(metrics)
            except Exception as e:
                logging.warning(f"Error evaluating example: {str(e)}")
                continue
                
        # Calculate average metrics
        avg_metrics = TranslationMetrics(
            bleu_score=np.mean([m.bleu_score for m in all_metrics]),
            rouge_1=np.mean([m.rouge_1 for m in all_metrics]),
            rouge_2=np.mean([m.rouge_2 for m in all_metrics]),
            rouge_l=np.mean([m.rouge_l for m in all_metrics]),
            bert_score_P=np.mean([m.bert_score_P for m in all_metrics]),
            bert_score_R=np.mean([m.bert_score_R for m in all_metrics]),
            bert_score_F=np.mean([m.bert_score_F for m in all_metrics])
        )
        
        return {
            'average_metrics': avg_metrics,
            'num_examples': len(all_metrics),
            'timestamp': datetime.now().isoformat()
        }
    
    def load_model(self, path: str) -> None:
        """Load a compiled model from file"""
        try:
            self.compiled_model = dspy.load(path)
            logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def load_and_validate_dataset(self, dataset_path: str) -> List[Example]:
        """
        Load and validate the translation dataset
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            List of validated translation examples
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            validated_examples = []
            for item in data:
                if self._validate_translation_pair(item):
                    validated_examples.append(Example(
                        english=item['english'],
                        spanish=item['spanish']
                    ).with_inputs('english'))
            
            logging.info(f"Loaded {len(validated_examples)} valid translation pairs")
            return validated_examples
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise
            
    def _validate_translation_pair(self, item: Dict) -> bool:
        """Validate a translation pair"""
        return (
            isinstance(item, dict) and
            'english' in item and
            'spanish' in item and
            isinstance(item['english'], str) and
            isinstance(item['spanish'], str) and
            len(item['english'].strip()) > 0 and
            len(item['spanish'].strip()) > 0
        )
    
    def _quality_check_wrapper(self, 
                           example: Example, 
                           pred: Example, 
                           trace: Optional[Dict] = None) -> float:
        """Wrapper for quality checking that includes multiple metrics"""
        metrics = self.evaluator.calculate_metrics(example.spanish, pred.spanish)
        
        # Define weights for selected metrics
        weights = {
            "bleu": 0.5,  
            "rouge_1": 0.2,  
            "rouge_2": 0.2,  
            "rouge_l": 0.1,
            "bert_score_P": 0.0,  
            "bert_score_R": 0.0,  
            "bert_score_F": 0.0   
        }

        # Compute weighted sum
        quality_score = (
            weights["bleu"] * metrics.bleu_score +
            weights["rouge_1"] * metrics.rouge_1 +
            weights["rouge_2"] * metrics.rouge_2 +
            weights["rouge_l"] * metrics.rouge_l +
            weights["bert_score_F"] * metrics.bert_score_F +
            weights["bert_score_P"] * metrics.bert_score_P +
            weights["bert_score_R"] * metrics.bert_score_R
        )

        return quality_score
    
    def translate_with_metrics(self, 
                             text: str, 
                             target: str = None,
                             return_metrics: bool = False) -> Union[str, Dict]:
        """
        Translate text and optionally return quality metrics
        
        Args:
            text: English text to translate
            target: Optional reference translation for metric calculation
            return_metrics: Whether to return quality metrics
            
        Returns:
            Translation string or dict with translation and metrics
        """
        if not self.compiled_model:
            raise ValueError("Model not compiled. Call compile_model first.")
            
        try:
            prediction = self.compiled_model(english=text)
            
            if not return_metrics:
                return prediction.spanish
                
            # Calculate metrics if reference translation available
            metrics = None
            if target:
                metrics = self.evaluator.calculate_metrics(
                    target,
                    prediction.spanish
                )
                
            return {
                'translation': prediction.spanish,
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            raise