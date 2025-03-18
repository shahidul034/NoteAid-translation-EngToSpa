from litgpt import LLM
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.current_device())  # Should print the selected GPU ID

llm = LLM.load("google/gemma-3-4b-it")
text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       