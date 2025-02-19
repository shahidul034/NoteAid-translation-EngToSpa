import dspy
from dataclasses import dataclass

@dataclass
class TranslationSignature(dspy.Signature):
    """Signature for English to Spanish translation"""
    english: str = dspy.InputField()
    spanish: str = dspy.OutputField()