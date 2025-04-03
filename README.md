# Result
## Phi-4 (Finetune) (gpt)


| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Back Avg BLEU | Back Avg chrF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------|----------------|------------------|---------------|---------------|-----------------|----------------|------------|
| Conceptual relationships extracted from the UMLS | 42.13 | 29.89 | 56.15 | 32.47 | 0.85996 | 0.85996 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 42.10 | 29.47 | 61.50 | 36.96 | 0.86270 | 0.86270 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 44.23 | 28.91 | 63.38 | 41.83 | 0.86299 | 0.86299 | 100 |
| Translation dictionary based on UMLS | 41.38 | 25.39 | 57.30 | 32.66 | 0.85409 | 0.85409 | 100 |
| Synonyms of each concept obtained from UMLS | 42.76 | 28.14 | 56.15 | 32.20 | 0.85026 | 0.85026 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 42.15 | 26.72 | 59.91 | 36.23 | 0.85488 | 0.85488 | 100 |
| Direct translation without context | 42.52 | 28.35 | 55.68 | 33.19 | 0.86159 | 0.86159 | 100 |  



## Phi-4 (Without finetune) (Alpaca)


| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Back Avg BLEU | Back Avg chrF+ | Direct Avg COMET | Back Avg COMET | Total Number of Data |
|------------------------|----------------|------------------|---------------|---------------|------------------|----------------|----------------------|
| Conceptual relationships extracted from the UMLS | 34.28 | 18.55 | 54.79 | 29.80 | 0.816 | 0.816 | 91 |
| Synonyms of each concept derived from GPT-4o Mini | 32.71 | 21.86 | 55.81 | 32.32 | 0.819 | 0.819 | 95 |
| Direct translation without context | 18.77 | 8.79 | 38.36 | 20.52 | 0.643 | 0.643 | 65 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 28.03 | 18.86 | 33.78 | 14.26 | 0.723 | 0.723 | 83 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 24.47 | 12.89 | 59.47 | 39.81 | 0.790 | 0.790 | 97 |
| Synonyms of each concept obtained from UMLS | 32.31 | 19.74 | 51.75 | 33.22 | 0.792 | 0.792 | 96 |
| Translation dictionary based on UMLS | 35.89 | 20.17 | 53.77 | 27.39 | 0.835 | 0.835 | 91 |


## Qwen2.5 14B (finetune) (gpt)


| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------|----------------|------------------|---------------|---------------|-----------------|----------------|------------|
| Direct translation without context | 38.63 | 23.53 | 52.89 | 30.43 | 0.8491 | 0.8491 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 41.95 | 25.93 | 38.44 | 28.92 | 0.8614 | 0.8614 | 100 |
| Synonyms of each concept obtained from UMLS | 39.71 | 24.76 | 44.04 | 28.51 | 0.8482 | 0.8482 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 39.17 | 23.82 | 41.62 | 31.05 | 0.8572 | 0.8572 | 100 |
| Conceptual relationships extracted from the UMLS | 40.37 | 24.87 | 30.56 | 21.35 | 0.8587 | 0.8587 | 100 |
| Translation dictionary based on UMLS | 39.47 | 26.28 | 13.61 | 14.63 | 0.8511 | 0.8511 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 39.91 | 22.97 | 50.84 | 30.84 | 0.8463 | 0.8463 | 100 |  




## Qwen2.5 14B (without finetune) (Alpaca)



| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------|----------------|------------------|---------------|---------------|------------------|---------------|------------|
| Translation dictionary based on UMLS | 34.18 | 21.77 | 54.09 | 32.39 | 0.8399 | 0.8399 | 100 |
| Conceptual relationships extracted from UMLS | 34.70 | 21.29 | 51.06 | 32.71 | 0.8491 | 0.8491 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 33.05 | 21.60 | 52.59 | 36.48 | 0.8420 | 0.8420 | 100 |
| Synonyms of each concept obtained from UMLS | 33.51 | 21.06 | 50.15 | 29.33 | 0.8323 | 0.8323 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 41.95 | 25.93 | 38.44 | 28.92 | 0.8614 | 0.8614 | 100 |
| Direct translation without context | 34.13 | 21.28 | 43.74 | 25.81 | 0.8314 | 0.8314 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 35.55 | 21.27 | 50.03 | 31.37 | 0.8438 | 0.8438 | 100 |  


## Meta-Llama-3.1-8B-Instruct (finetune) (Alpaca)


| Contextual Information                                   | Direct Avg BLEU | Direct Avg CHRF+ | Back Avg BLEU | Back Avg CHRF+ | Direct Avg COMET | Back Avg COMET |
|----------------------------------------------------------|----------------|-----------------|---------------|----------------|-----------------|----------------|
| synonyms of each concept derived from GPT-4o Mini             | 33.12          | 23.25           | 54.86         | 38.17          | 0.8526          | 0.8526         |
| Translation dictionary based on UMLS                | 34.13          | 21.10           | 54.43         | 34.39          | 0.8483          | 0.8483         |
| Knowledge graphs of each concept from GPT-4o Mini   | 35.62          | 22.33           | 51.45         | 33.79          | 0.8419          | 0.8419         |
| conceptual relationships extracted from the UMLS           | 33.27          | 21.09           | 48.62         | 27.41          | 0.8425          | 0.8425         |
| Direct translation without context                           | 34.47          | 22.11           | 48.59         | 29.96          | 0.8502          | 0.8502         |
| Multilingual translations of each concept from GPT-4o Mini | 33.15          | 19.67           | 51.05         | 38.42          | 0.8481          | 0.8481         |
| synonyms of each concept obtained from UMLS | 32.52          | 21.83           | 51.25         | 30.14          | 0.8357          | 0.8357         |




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

 ## Qwen2.5 3B (finetune) (gpt)

| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Back Avg BLEU | Back Avg chrF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------|----------------|------------------|---------------|---------------|------------------|---------------|------------|
| Direct translation without context | 38.83 | 22.33 | 49.35 | 26.30 | 0.856 | 0.856 | 99 |
| Translation dictionary based on UMLS | 38.76 | 22.85 | 9.69 | 10.19 | 0.852 | 0.852 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 37.83 | 23.72 | 38.42 | 21.71 | 0.854 | 0.854 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 38.41 | 22.47 | 9.77 | 10.33 | 0.856 | 0.856 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 40.88 | 24.61 | 39.57 | 23.03 | 0.851 | 0.851 | 99 |
| Conceptual relationships extracted from the UMLS | 36.31 | 22.78 | 18.88 | 14.53 | 0.852 | 0.852 | 99 |  

  ## Qwen2.5 3B (without finetune)

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Back Avg BLEU | Back Avg chrF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|---------------|---------------|------------------|----------------|------------|
| Direct translation without context                        | 29.10           | 17.99            | 40.81         | 23.39         | 0.8262           | 0.8262         | 100        |
| Synonyms of each concept obtained from UMLS               | 21.73           | 12.69            | 21.34         | 10.14         | 0.7416           | 0.7416         | 98         |
| Translation dictionary based on UMLS                      | 22.73           | 13.52            | 30.71         | 18.46         | 0.7485           | 0.7485         | 99         |
| Synonyms of each concept derived from GPT-4o Mini        | 28.31           | 19.44            | 41.82         | 21.54         | 0.8225           | 0.8225         | 99         |
| Conceptual relationships extracted from the UMLS         | 23.41           | 13.51            | 32.86         | 17.86         | 0.7679           | 0.7679         | 99         |
| Knowledge graphs of each concept generated using GPT-4o Mini | 24.14           | 12.07            | 35.25         | 20.93         | 0.7973           | 0.7973         | 98         |
| Multilingual translations of each concept obtained from GPT-4o Mini | 27.25           | 18.59            | 41.15         | 22.00         | 0.8175           | 0.8175         | 99         |

 ## Qwen2.5 1.5B (finetune) (gpt)


| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Back Avg BLEU | Back Avg chrF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|---------------|---------------|------------------|----------------|------------|
| Synonyms of each concept derived from GPT-4o Mini        | 26.71           | 18.53            | 37.77         | 23.59         | 0.8279           | 0.8279         | 100        |
| Multilingual translations of each concept obtained from GPT-4o Mini | 27.72           | 19.78            | 34.38         | 21.07         | 0.8350           | 0.8350         | 100        |
| Direct translation without context                        | 29.01           | 20.52            | 36.66         | 23.32         | 0.8373           | 0.8373         | 100        |
| Translation dictionary based on UMLS                      | 26.67           | 19.56            | 32.03         | 21.18         | 0.8313           | 0.8313         | 98         |
| Synonyms of each concept obtained from UMLS               | 25.13           | 19.03            | 33.63         | 19.66         | 0.8234           | 0.8234         | 99         |
| Conceptual relationships extracted from the UMLS         | 26.99           | 20.47            | 34.30         | 24.31         | 0.8247           | 0.8247         | 99         |
| Knowledge graphs of each concept generated using GPT-4o Mini | 25.10           | 16.97            | 35.36         | 23.29         | 0.8131           | 0.8131         | 100        |

## Qwen2.5 1.5B (without finetune)

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Back Avg BLEU | Back Avg chrF+ | Direct Avg COMET | Back Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|---------------|---------------|------------------|----------------|------------|
| Direct translation without context                        | 26.44           | 14.51            | 36.56         | 23.07         | 0.8174           | 0.8174         | 100        |
| Synonyms of each concept derived from GPT-4o Mini        | 24.57           | 17.68            | 40.09         | 20.65         | 0.8149           | 0.8149         | 100        |
| Multilingual translations of each concept obtained from GPT-4o Mini | 25.57           | 18.71            | 38.76         | 20.93         | 0.8253           | 0.8253         | 100        |
| Translation dictionary based on UMLS                      | 26.05           | 19.83            | 36.73         | 20.58         | 0.8104           | 0.8104         | 100        |
| Conceptual relationships extracted from the UMLS         | 25.32           | 17.10            | 35.66         | 21.17         | 0.8211           | 0.8211         | 99         |
| Knowledge graphs of each concept generated using GPT-4o Mini | 22.19           | 13.56            | 34.31         | 19.31         | 0.7983           | 0.7983         | 99         |
| Synonyms of each concept obtained from UMLS               | 23.12           | 14.07            | 33.36         | 17.48         | 0.7954           | 0.7954         | 100        |


