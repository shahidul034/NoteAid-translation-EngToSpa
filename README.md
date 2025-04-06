# Result
## Phi-4 (Finetune) (gpt)

| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|-----------------|------------|
| Conceptual relationships extracted from the UMLS | 42.13 | 29.89 | 0.85996 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 42.10 | 29.47 | 0.86270 | 100 |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 44.23 | 28.91 | 0.86299 | 100 |
| Translation dictionary based on UMLS | 41.38 | 25.39 | 0.85409 | 100 |
| Synonyms of each concept obtained from UMLS | 42.76 | 28.14 | 0.85026 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 42.15 | 26.72 | 0.85488 | 100 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 42.52 | 28.35 | 0.86159 | 100 |


## Phi-4 (Without Finetune) (Alpaca)

| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|-----------------|------------|
| Conceptual relationships extracted from the UMLS | 34.28 | 18.55 | 0.816 | 91 |
| Synonyms of each concept derived from GPT-4o Mini | 32.71 | 21.86 | 0.819 | 95 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 28.03 | 18.86 | 0.723 | 83 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 24.47 | 12.89 | 0.790 | 97 |
| Synonyms of each concept obtained from UMLS | 32.31 | 19.74 | 0.792 | 96 |
| ✅ Translation dictionary based on UMLS | 35.89 | 20.17 | 0.835 | 91 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 18.77 | 8.79 | 0.643 | 65 |

## Qwen2.5 14B (Finetuned) (GPT)

| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|-----------------|------------|
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 41.95 | 25.93 | 0.8614 | 100 |
| Synonyms of each concept obtained from UMLS | 39.71 | 24.76 | 0.8482 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 39.17 | 23.82 | 0.8572 | 100 |
| Conceptual relationships extracted from the UMLS | 40.37 | 24.87 | 0.8587 | 100 |
| Translation dictionary based on UMLS | 39.47 | 26.28 | 0.8511 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 39.91 | 22.97 | 0.8463 | 100 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 38.63 | 23.53 | 0.8491 | 100 |


## Qwen2.5 14B (without finetune) (Alpaca)  

| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|------------------|------------|
| Translation dictionary based on UMLS | 34.18 | 21.77 | 0.8399 | 100 |
| Conceptual relationships extracted from UMLS | 34.70 | 21.29 | 0.8491 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 33.05 | 21.60 | 0.8420 | 100 |
| Synonyms of each concept obtained from UMLS | 33.51 | 21.06 | 0.8323 | 100 |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 41.95 | 25.93 | 0.8614 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 35.55 | 21.27 | 0.8438 | 100 |  
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 34.13 | 21.28 | 0.8314 | 100 |


## Meta-Llama-3.1-8B-Instruct (finetune) (Alpaca)  

| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Direct Avg COMET |
|----------------------------------------------------------|----------------|-----------------|-----------------|
| ✅  Synonyms of each concept derived from GPT-4o Mini | 33.12 | 23.25 | 0.8526 |
| Translation dictionary based on UMLS | 34.13 | 21.10 | 0.8483 |
| Knowledge graphs of each concept from GPT-4o Mini | 35.62 | 22.33 | 0.8419 |
| Conceptual relationships extracted from the UMLS | 33.27 | 21.09 | 0.8425 |
| Multilingual translations of each concept from GPT-4o Mini | 33.15 | 19.67 | 0.8481 |
| Synonyms of each concept obtained from UMLS | 32.52 | 21.83 | 0.8357 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 34.47 | 22.11 | 0.8502 |


## Meta-Llama-3.1-8B-Instruct (without finetune) (Alpaca)  

| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|------------------|------------|
| ✅ Synonyms of each concept derived from GPT-4o Mini | 30.224 | 20.07 | 0.798 | 100 |
| Translation dictionary based on UMLS | 27.54 | 19.23 | 0.7921 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 25.99 | 15.93 | 0.7458 | 100 |
| Conceptual relationships extracted from the UMLS | 27.03 | 18.16 | 0.7760 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 28.637 | 21.84 | 0.8083 | 100 |
| Synonyms of each concept obtained from UMLS | 29.37 | 19.69 | 0.7949 | 100 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 29.36 | 18.94 | 0.8113 | 100 |


## Qwen2.5 7B (finetune) (gpt)  

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|------------------|------------|
| Translation dictionary based on UMLS                      | 38.76           | 23.63            | 0.8529           | 98         |
| Synonyms of each concept derived from GPT-4o Mini        | 38.43           | 24.05            | 0.8580           | 97         |
| Multilingual translations of each concept obtained from GPT-4o Mini | 38.86           | 22.80            | 0.8565           | 100        |
| ✅ Knowledge graphs of each concept generated using GPT-4o Mini | 40.47           | 24.58            | 0.8487           | 96         |
| Synonyms of each concept obtained from UMLS               | 38.24           | 23.40            | 0.8449           | 100        |
| Conceptual relationships extracted from the UMLS         | 38.83           | 24.35            | 0.8525           | 94         |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                        | 39.09           | 23.94            | 0.8555           | 96         |


## Qwen2.5 7B (without finetune) (gpt)  

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|------------------|------------|
| Conceptual relationships extracted from the UMLS         | 25.30           | 16.59            | 0.7598           | 100        |
| Knowledge graphs of each concept generated using GPT-4o Mini | 26.85           | 18.01            | 0.7717           | 100        |
| Synonyms of each concept obtained from UMLS               | 19.29           | 10.88            | 0.6970           | 95         |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 33.40           | 20.35            | 0.8451           | 100        |
| Synonyms of each concept derived from GPT-4o Mini        | 31.07           | 19.14            | 0.8211           | 99         |
| Translation dictionary based on UMLS                      | 24.50           | 16.58            | 0.7607           | 99         |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                        | 31.77           | 21.54            | 0.8402           | 100        |

---

## Qwen2.5 3B (finetune) (gpt)  

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|------------------|------------|
| Translation dictionary based on UMLS                      | 38.76           | 22.85            | 0.852            | 100        |
| Synonyms of each concept derived from GPT-4o Mini        | 37.83           | 23.72            | 0.854            | 100        |
| Multilingual translations of each concept obtained from GPT-4o Mini | 38.41           | 22.47            | 0.856            | 100        |
| ✅ Knowledge graphs of each concept generated using GPT-4o Mini | 40.88           | 24.61            | 0.851            | 99         |
| Conceptual relationships extracted from the UMLS         | 36.31           | 22.78            | 0.852            | 99         |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                        | 38.83           | 22.33            | 0.856            | 99         |


## Qwen2.5 3B (without finetune) ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) 

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|------------------|------------|
| Synonyms of each concept obtained from UMLS               | 21.73           | 12.69            | 0.7416           | 98         |
| Translation dictionary based on UMLS                      | 22.73           | 13.52            | 0.7485           | 99         |
| Synonyms of each concept derived from GPT-4o Mini        | 28.31           | 19.44            | 0.8225           | 99         |
| Conceptual relationships extracted from the UMLS         | 23.41           | 13.51            | 0.7679           | 99         |
| Knowledge graphs of each concept generated using GPT-4o Mini | 24.14           | 12.07            | 0.7973           | 98         |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 29.73           | 17.30            | 0.838           | 99         |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                        | 28.62           | 17.19            | 0.8274           | 100        |


## Qwen2.5 1.5B (finetune) (gpt) 


| Contextual Information                                         | Direct Avg BLEU | Direct Avg chrF++ | Direct Avg COMET | Total Number of Data |
|----------------------------------------------------------------|------------------|--------------------|-------------------|------------------------|
| Synonyms of each concept derived from GPT-4o Mini              | 26.78            | 18.13              | 0.8308            | 100                    |
| Synonyms of each concept obtained from UMLS                    | 23.73            | 16.39              | 0.8226            | 100                    |
| Translation dictionary based on UMLS                           | 24.73            | 17.08              | 0.8256            | 100                    |
| Knowledge graphs of each concept generated using GPT-4o Mini   | 25.08            | 17.27              | 0.8154            | 100                    |
| Direct translation without context                             | 27.61            | 19.98              | 0.8306            | 100                    |
| ✅ Multilingual translations of each concept from GPT-4o Mini     | 27.83            | 19.01              | 0.8376            | 100                    |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Conceptual relationships extracted from the UMLS               | 26.30            | 18.09              | 0.8273            | 100                    |


## Qwen2.5 1.5B (without finetune)  


| Contextual Information                                         | Direct Avg BLEU | Direct Avg chrF++ | Direct Avg COMET | Total Number of Data |
|----------------------------------------------------------------|------------------|--------------------|-------------------|------------------------|
| Synonyms of each concept derived from GPT-4o Mini              | 26.78            | 18.13              | 0.8308            | 100                    |
| Synonyms of each concept obtained from UMLS                    | 23.73            | 16.39              | 0.8226            | 100                    |
| Translation dictionary based on UMLS                           | 24.73            | 17.08              | 0.8256            | 100                    |
| Knowledge graphs of each concept generated using GPT-4o Mini   | 25.08            | 17.27              | 0.8154            | 100                    |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                             | 27.61            | 19.98              | 0.8306            | 100                    |
| ✅ Multilingual translations of each concept from GPT-4o Mini     | 27.83            | 19.01              | 0.8376            | 100                    |
| Conceptual relationships extracted from the UMLS               | 26.30            | 18.09              | 0.8273            | 100                    |



