import dspy
from signatures.TranslationSignature import TranslationSignature

# class TranslationModule(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.generate_answer = dspy.ChainOfThought(TranslationSignature)
    
#     def forward(self, original, lang):
#         prediction = self.generate_answer(original=original, lang=lang)
#         return dspy.Prediction(spanish=prediction.spanish)

class TranslationModule(dspy.Module):
    """Module for handling English to Spanish translation"""
    def __init__(self):
        super().__init__()
        self.translator = dspy.Predict(TranslationSignature)
    
    def forward(self, english):
        """Translate English text to Spanish"""
        return self.translator(english=english)