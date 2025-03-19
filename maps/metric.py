from dataclasses import dataclass

@dataclass
class TranslationMetrics:
    """Store multiple translation quality metrics"""
    bleu_score: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bert_score_P: float
    bert_score_R: float
    bert_score_F: float
    
    def __str__(self):
        return f"BLEU: {self.bleu_score:.2f}, ROUGE-1: {self.rouge_1:.2f}, ROUGE-2: {self.rouge_2:.2f}, ROUGE-L: {self.rouge_l:.2f}, BERT-P: {self.bert_score_P:.2f}, BERT-R: {self.bert_score_R:.2f}, BERT-F: {self.bert_score_F:.2f}"
