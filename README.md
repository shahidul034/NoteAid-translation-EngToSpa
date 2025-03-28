# Result
## Phi-4 (Finetune) (gpt)

| Contextual Information                                           | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ |
|------------------------------------------------------------------|----------------|-----------------|---------------|---------------|
| Conceptual relationships extracted from the UMLS                | 42.13          | 29.89           | 56.15         | 32.47         |
| Synonyms of each concept derived from GPT-4o Mini               | 42.10          | 29.47           | 61.50         | 36.96         |
| Multilingual translations of each concept obtained from GPT-4o Mini | 44.23          | 28.91           | 63.38         | 41.83         |
| Translation dictionary based on UMLS                            | 41.38          | 25.39           | 57.30         | 32.66         |
| Synonyms of each concept obtained from UMLS                     | 42.76          | 28.14           | 56.15         | 32.20         |
| Knowledge graphs of each concept generated using GPT-4o Mini    | 42.15          | 26.72           | 59.91         | 36.23         |
| Direct translation without context                              | 42.52          | 28.35           | 55.68         | 33.19         |
