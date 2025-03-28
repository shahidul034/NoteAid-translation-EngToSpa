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

## Phi-4 (Without finetune) (Alpaca)

| Contextual Information                                           | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ |
|------------------------------------------------------------------|----------------|-----------------|---------------|---------------|
| Conceptual relationships extracted from the UMLS                | 32.95          | 17.84           | 49.59         | 27.35         |
| Synonyms of each concept derived from GPT-4o Mini               | 32.09          | 21.22           | 50.37         | 30.65         |
| Direct translation without context                              | 12.06          | 7.02            | 23.75         | 14.20         |
| Multilingual translations of each concept obtained from GPT-4o Mini | 23.62          | 12.61           | 56.14         | 38.75         |
| Synonyms of each concept obtained from UMLS                     | 31.01          | 19.11           | 48.44         | 32.07         |
| Translation dictionary based on UMLS                            | 35.22          | 20.19           | 48.48         | 25.17         |
| Knowledge graphs of each concept generated using GPT-4o Mini    | 24.43          | 16.74           | 27.82         | 12.73         |
