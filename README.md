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




## Qwen2.5 14B (finetune) (gpt)

| Contextual Information                                      | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Total Data |
|-------------------------------------------------------------|-----------------|------------------|---------------|---------------|------------|
| Direct translation without context                         | 38.63           | 23.53            | 52.89         | 30.43         | 100        |
| Multilingual translations of each concept (GPT-4o Mini)    | 41.95           | 25.93            | 38.44         | 28.92         | 100        |
| Synonyms of each concept obtained from UMLS                | 39.71           | 24.76            | 44.04         | 28.51         | 100        |
| Synonyms of each concept derived from GPT-4o Mini         | 39.17           | 23.82            | 41.62         | 31.05         | 100        |
| Conceptual relationships extracted from UMLS              | 40.37           | 24.87            | 30.56         | 21.35         | 100        |
| Translation dictionary based on UMLS                      | 39.47           | 26.28            | 13.61         | 14.63         | 100        |
| Knowledge graphs of each concept (GPT-4o Mini)           | 39.91           | 22.97            | 50.84         | 30.84         | 100        |




## Qwen2.5 14B (without finetune) (Alpaca)

| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Total Data |
|------------------------|----------------|------------------|---------------|---------------|------------|
| Translation dictionary based on UMLS | 33.75 | 21.77 | 54.09 | 32.39 | 100 |
| Conceptual relationships extracted from the UMLS | 34.70 | 21.29 | 51.03 | 32.71 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 32.48 | 21.59 | 52.41 | 36.48 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 34.26 | 23.38 | 56.18 | 33.82 | 100 |
| Synonyms of each concept obtained from UMLS | 33.14 | 21.06 | 49.71 | 29.33 | 100 |
| Direct translation without context | 33.15 | 21.25 | 43.19 | 25.81 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 35.43 | 21.27 | 49.54 | 31.37 | 100 |

## Meta-Llama-3.1-8B-Instruct (finetune) (Alpaca)


| Contextual Information                                   | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Direct Avg COMET | Back Avg COMET |
|----------------------------------------------------------|----------------|-----------------|---------------|----------------|-----------------|----------------|
| synonyms of each concept derived from GPT-4o Mini             | 33.12          | 23.25           | 54.86         | 38.17          | 0.8526          | 0.8526         |
| Translation dictionary based on UMLS                | 34.13          | 21.10           | 54.43         | 34.39          | 0.8483          | 0.8483         |
| Knowledge graphs of each concept from GPT-4o Mini   | 35.62          | 22.33           | 51.45         | 33.79          | 0.8419          | 0.8419         |
| conceptual relationships extracted from the UMLS           | 33.27          | 21.09           | 48.62         | 27.41          | 0.8425          | 0.8425         |
| Direct translation without context                           | 34.47          | 22.11           | 48.59         | 29.96          | 0.8502          | 0.8502         |
| Multilingual translations of each concept from GPT-4o Mini | 33.15          | 19.67           | 51.05         | 38.42          | 0.8481          | 0.8481         |
| synonyms of each concept obtained from UMLS (Meta-Llama-3.1-8B-Instruct_alpaca_alpaca) | 32.52          | 21.83           | 51.25         | 30.14          | 0.8357          | 0.8357         |




## Meta-Llama-3.1-8B-Instruct (without finetune) (Alpaca)


| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Direct Avg COMET | Back Avg COMET | Total Number of Data |
|------------------------|----------------|------------------|--------------|--------------|----------------|----------------|------------------|
| Synonyms of each concept derived from GPT-4o Mini | 27.11 | 20.07 | 42.08 | 33.27 | 0.7528 | 0.7528 | 100 |
| Translation dictionary based on UMLS | 27.54 | 19.23 | 47.32 | 36.92 | 0.7921 | 0.7921 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 25.99 | 15.93 | 43.19 | 32.97 | 0.7458 | 0.7458 | 100 |
| Conceptual relationships extracted from the UMLS | 27.03 | 18.16 | 43.90 | 36.33 | 0.7760 | 0.7760 | 100 |
| Direct translation without context | 29.36 | 18.94 | 42.87 | 28.21 | 0.8113 | 0.8113 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 27.70 | 20.11 | 44.52 | 40.09 | 0.7871 | 0.7871 | 100 |
| Synonyms of each concept obtained from UMLS | 29.13 | 17.56 | 46.14 | 29.83 | 0.7915 | 0.7915 | 100 |


