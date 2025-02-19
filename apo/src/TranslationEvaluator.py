
from src.TranslationMetrics import TranslationMetrics

import logging
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import sacrebleu

class TranslationEvaluator:
    """Handles translation quality evaluation using multiple metrics for Spanish text"""
    
    def __init__(self):
        self.res = {}
        self.data = None
        self.target = None
        self.generated = None
        # # Initialize BERTScorer once during class initialization
        # self.bert_scorer = BERTScorer(lang="sp", rescale_with_baseline=True)
        
    def bleu(self, hypothesis: str, reference: str) -> float:
        """Calculates BLEU score using sacrebleu's Python API and returns the score"""
        try:
            bleu_score = sacrebleu.corpus_bleu([hypothesis], [[reference]]).score / 100.0
            return bleu_score
        except Exception as e:
            logging.error(f"Error calculating BLEU score: {str(e)}")
            return 0.0

    def calculate_metrics(self, reference: str, translation: str) -> TranslationMetrics:
        """
        Calculate multiple translation quality metrics for Spanish text
        
        Args:
            reference: Reference text in Spanish
            translation: Model's translated text
            
        Returns:
            TranslationMetrics object containing various scores
        """
        try:
            # Set up the data for helper methods
            self.target = [reference]
            self.generated = [translation]
            self.res = {}
            
            self._rouge()
            rouge_1_f = self.res['rouge-1'][2]  
            rouge_2_f = self.res['rouge-2'][2]
            rouge_l_f = self.res['rouge-l'][2]
            
            bleu_score = self.bleu(translation, reference)

            # bert_P, bert_R, bert_F = self.bert_scorer.score(self.generated, self.target)

            # Create metrics object
            return TranslationMetrics(
                bleu_score=bleu_score,
                rouge_1=rouge_1_f,
                rouge_2=rouge_2_f,
                rouge_l=rouge_l_f,
                bert_score_P=0.0,
                bert_score_R=0.0,
                bert_score_F=0.0
                # bert_score_P=bert_P.mean().item(),
                # bert_score_R=bert_R.mean().item(),
                # bert_score_F=bert_F.mean().item()
            )
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return TranslationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _rouge(self):
        """Calculate ROUGE scores"""
        r1 = {'p': [], 'r': [], 'f': []}
        r2 = {'p': [], 'r': [], 'f': []}
        rl = {'p': [], 'r': [], 'f': []}
        
        rouge_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        for i in range(len(self.generated)):
            all_rouge_scores = rouge_instance.score(self.target[i], self.generated[i])
            r1['p'].append(all_rouge_scores['rouge1'][0])
            r1['r'].append(all_rouge_scores['rouge1'][1])
            r1['f'].append(all_rouge_scores['rouge1'][2])

            r2['p'].append(all_rouge_scores['rouge2'][0])
            r2['r'].append(all_rouge_scores['rouge2'][1])
            r2['f'].append(all_rouge_scores['rouge2'][2])

            rl['p'].append(all_rouge_scores['rougeL'][0])
            rl['r'].append(all_rouge_scores['rougeL'][1])
            rl['f'].append(all_rouge_scores['rougeL'][2])

        self.res['rouge-1'] = (sum(r1['p'])/len(r1['p']), sum(r1['r'])/len(r1['r']), sum(r1['f'])/len(r1['f']))
        self.res['rouge-2'] = (sum(r2['p'])/len(r2['p']), sum(r2['r'])/len(r2['r']), sum(r2['f'])/len(r2['f']))
        self.res['rouge-l'] = (sum(rl['p'])/len(rl['p']), sum(rl['r'])/len(rl['r']), sum(rl['f'])/len(rl['f']))
