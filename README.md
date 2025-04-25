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

## Significant testing

| **Context Information**                                        | **Metric**         | **Mean (μ)** | **Std Dev (σ)** | **Std Error (SE)** | **95% CI**                  |
|----------------------------------------------------------------|--------------------|--------------|------------------|--------------------|------------------------------|
| Synonyms (UMLS)                                                | BLEU               | 42.5080      | 0.0000           | 0.0000             | (42.5080, 42.5080)           |
|                                                                | CHRF++             | 27.2871      | 0.0000           | 0.0000             | (27.2871, 27.2871)           |
|                                                                | COMET              | 0.8497       | 0.0000           | 0.0000             | (0.8497, 0.8497)             |
| Knowledge Graphs (GPT-4o Mini)                                 | BLEU               | 42.5755      | 0.0000           | 0.0000             | (42.5755, 42.5755)           |
|                                                                | CHRF++             | 27.4187      | 0.0000           | 0.0000             | (27.4187, 27.4187)           |
|                                                                | COMET              | 0.8548       | 0.0000           | 0.0000             | (0.8548, 0.8548)             |
| Translation Dictionary (UMLS)                                  | BLEU               | 40.2544      | 0.0000           | 0.0000             | (40.2544, 40.2544)           |
|                                                                | CHRF++             | 25.3336      | 0.0000           | 0.0000             | (25.3336, 25.3336)           |
|                                                                | COMET              | 0.8508       | 0.0000           | 0.0000             | (0.8508, 0.8508)             |
| Conceptual Relationships (UMLS)                                | BLEU               | **44.1976**  | 0.0000           | 0.0000             | (44.1976, 44.1976)           |
|                                                                | CHRF++             | **30.1960**  | 0.0000           | 0.0000             | (30.1960, 30.1960)           |
|                                                                | COMET              | **0.8614**   | 0.0000           | 0.0000             | (0.8614, 0.8614)             |
| Synonyms (GPT-4o Mini)                                         | BLEU               | 41.3946      | 0.0000           | 0.0000             | (41.3946, 41.3946)           |
|                                                                | CHRF++             | 28.4210      | 0.0000           | 0.0000             | (28.4210, 28.4210)           |
|                                                                | COMET              | 0.8605       | 0.0000           | 0.0000             | (0.8605, 0.8605)             |
| Direct Translation (No Context)                                | BLEU               | 42.7264      | 0.0000           | 0.0000             | (42.7264, 42.7264)           |
|                                                                | CHRF++             | 27.3038      | 0.0000           | 0.0000             | (27.3038, 27.3038)           |
|                                                                | COMET              | 0.8600       | 0.0000           | 0.0000             | (0.8600, 0.8600)             |
| Multilingual Translations (GPT-4o Mini)                        | BLEU               | 43.3109      | 0.0000           | 0.0000             | (43.3109, 43.3109)           |
|                                                                | CHRF++             | 27.9407      | 0.0000           | 0.0000             | (27.9407, 27.9407)           |
|                                                                | COMET              | **0.8614**   | 0.0000           | 0.0000             | (0.8614, 0.8614)             |

---

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
---
## Significant testing


| **Context Information**                                               | **Metric**         | **Mean (μ)** | **Std Dev (σ)** | **Std Error (SE)** | **95% CI**                  |
|------------------------------------------------------------------------|--------------------|--------------|------------------|--------------------|------------------------------|
| Translation dictionary based on UMLS                                   | BLEU               | 36.4711      | 0.0895           | 0.0400             | (36.3926, 36.5495)           |
|                                                                        | CHRF++             | 20.7827      | 0.0630           | 0.0282             | (20.7275, 20.8379)           |
|                                                                        | COMET              | 0.8325       | 0.0013           | 0.0006             | (0.8314, 0.8337)             |
| Conceptual relationships extracted from the UMLS                       | BLEU               | 34.8318      | 0.1275           | 0.0570             | (34.7200, 34.9436)           |
|                                                                        | CHRF++             | 18.9465      | 0.1150           | 0.0514             | (18.8457, 19.0473)           |
|                                                                        | COMET              | 0.8185       | 0.0017           | 0.0007             | (0.8170, 0.8199)             |
| Direct translation without context                                     | BLEU               | 18.9240      | 0.1409           | 0.0630             | (18.8005, 19.0475)           |
|                                                                        | CHRF++             | 9.1569       | 0.2029           | 0.0907             | (8.9790, 9.3347)             |
|                                                                        | COMET              | 0.6341       | 0.0052           | 0.0023             | (0.6296, 0.6387)             |
| Synonyms of each concept obtained from UMLS                            | BLEU               | 32.8933      | 0.3267           | 0.1461             | (32.6069, 33.1797)           |
|                                                                        | CHRF++             | 19.7372      | 0.0027           | 0.0012             | (19.7348, 19.7395)           |
|                                                                        | COMET              | 0.7965       | 0.0025           | 0.0011             | (0.7944, 0.7987)             |
| Multilingual translations of each concept obtained from GPT-4o Mini    | BLEU               | 24.6117      | 0.0765           | 0.0342             | (24.5447, 24.6788)           |
|                                                                        | CHRF++             | 12.6683      | 0.1240           | 0.0555             | (12.5595, 12.7770)           |
|                                                                        | COMET              | 0.7880       | 0.0011           | 0.0005             | (0.7870, 0.7890)             |
| Knowledge graphs of each concept generated using GPT-4o Mini           | BLEU               | 28.4230      | 0.0030           | 0.0014             | (28.4203, 28.4256)           |
|                                                                        | CHRF++             | 19.0864      | 0.0903           | 0.0404             | (19.0073, 19.1655)           |
|                                                                        | COMET              | 0.7258       | 0.0013           | 0.0006             | (0.7247, 0.7270)             |
| Synonyms of each concept derived from GPT-4o Mini                      | BLEU               | 32.0276      | 0.6026           | 0.2695             | (31.4994, 32.5558)           |
|                                                                        | CHRF++             | 20.9581      | 0.5116           | 0.2288             | (20.5097, 21.4065)           |
|                                                                        | COMET              | 0.8142       | 0.0024           | 0.0011             | (0.8121, 0.8164)             |


## Qwen2.5 14B (Finetune) (GPT)

| Contextual Information | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|-----------------|------------|
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 41.95 | 25.93 | 0.8614 | 100 |
| Synonyms of each concept obtained from UMLS | 39.71 | 24.76 | 0.8482 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 39.17 | 23.82 | 0.8572 | 100 |
| Conceptual relationships extracted from the UMLS | 40.37 | 24.87 | 0.8587 | 100 |
| Translation dictionary based on UMLS | 39.47 | 26.28 | 0.8511 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 39.91 | 22.97 | 0.8463 | 100 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 38.63 | 23.53 | 0.8491 | 100 |

## Significant testing

| **Context**                                                      | **Metric** | **Mean**  | **Std Dev** | **Std Error** | **CI Lower** | **CI Upper** |
|------------------------------------------------------------------|------------|-----------|-------------|----------------|--------------|--------------|
| Direct translation without context                               | BLEU       | 35.3830   | 0.1300      | 0.0581         | 35.2691      | 35.4970      |
|                                                                  | CHRF+      | 18.5851   | 0.1989      | 0.0890         | 18.4107      | 18.7595      |
|                                                                  | COMET      | 0.8563    | 0.0001      | 0.0001         | 0.8562       | 0.8564       |
| Synonyms of each concept obtained from UMLS                      | BLEU       | 38.2293   | 0.0171      | 0.0076         | 38.2144      | 38.2443      |
|                                                                  | CHRF+      | 23.3924   | 0.0084      | 0.0038         | 23.3850      | 23.3998      |
|                                                                  | COMET      | 0.8460    | 0.0000      | 0.0000         | 0.8460       | 0.8460       |
| Translation dictionary based on UMLS                             | BLEU       | 35.4518   | 0.4980      | 0.2227         | 35.0153      | 35.8883      |
|                                                                  | CHRF+      | 21.6181   | 0.4335      | 0.1939         | 21.2381      | 21.9981      |
|                                                                  | COMET      | 0.8496    | 0.0020      | 0.0009         | 0.8478       | 0.8514       |
| Multilingual translations of each concept obtained from GPT-4o Mini | BLEU    | 37.8983   | 0.2962      | 0.1324         | 37.6387      | 38.1578      |
|                                                                  | CHRF+      | 23.5851   | 0.1394      | 0.0623         | 23.4629      | 23.7073      |
|                                                                  | COMET      | 0.8588    | 0.0003      | 0.0001         | 0.8585       | 0.8590       |
| Knowledge graphs of each concept generated using GPT-4o Mini     | BLEU       | 37.9855   | 0.0682      | 0.0305         | 37.9258      | 38.0453      |
|                                                                  | CHRF+      | 21.7686   | 0.0131      | 0.0059         | 21.7571      | 21.7802      |
|                                                                  | COMET      | 0.8492    | 0.0001      | 0.0000         | 0.8491       | 0.8492       |
| Conceptual relationships extracted from the UMLS                 | BLEU       | 37.5785   | 0.0535      | 0.0239         | 37.5317      | 37.6254      |
|                                                                  | CHRF+      | 22.4503   | 0.0890      | 0.0398         | 22.3723      | 22.5283      |
|                                                                  | COMET      | 0.8539    | 0.0001      | 0.0001         | 0.8538       | 0.8540       |


---
## Qwen2.5 14B (without finetune) (Alpaca)  

| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|------------------|------------|
| Translation dictionary based on UMLS | 34.18 | 21.77 | 0.8399 | 100 |
| Conceptual relationships extracted from UMLS | 34.70 | 21.29 | 0.8491 | 100 |
| Synonyms of each concept derived from GPT-4o Mini | 33.05 | 21.60 | 0.8420 | 100 |
| Synonyms of each concept obtained from UMLS | 33.51 | 21.06 | 0.8323 | 100 |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 41.95 | 25.93 | 0.8614 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 35.55 | 21.27 | 0.8438 | 100 |  
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 31.23 | 18.012 | 0.83 | 100 |

## Significant testing

| **Context Information**                                               | **Metric** | **Mean (μ)** | **Standard Deviation (σ)** | **Standard Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|-----------------------------|--------------------------|------------------------------------------|
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 30.3440      | 0.1357                      | 0.0607                   | (30.2251, 30.4630)                        |
|                                                                        | chrF++     | 19.7602      | 0.3779                      | 0.1690                   | (19.4290, 20.0914)                        |
|                                                                        | COMET      | 0.8152       | 0.0008                      | 0.0004                   | (0.8145, 0.8159)                          |
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 34.7523      | 0.1121                      | 0.0501                   | (34.6541, 34.8505)                        |
|                                                                        | chrF++     | 22.1956      | 0.4191                      | 0.1874                   | (21.8283, 22.5629)                        |
|                                                                        | COMET      | 0.8327       | 0.0020                      | 0.0009                   | (0.8309, 0.8344)                          |
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 27.0515      | 0.4993                      | 0.2233                   | (26.6138, 27.4891)                        |
|                                                                        | chrF++     | 15.2687      | 0.4857                      | 0.2172                   | (14.8430, 15.6945)                        |
|                                                                        | COMET      | 0.7538       | 0.0030                      | 0.0014                   | (0.7512, 0.7565)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 31.5381      | 0.0909                      | 0.0407                   | (31.4584, 31.6178)                        |
|                                                                        | chrF++     | 16.8554      | 0.4368                      | 0.1954                   | (16.4725, 17.2383)                        |
|                                                                        | COMET      | 0.8214       | 0.0031                      | 0.0014                   | (0.8187, 0.8241)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 31.5182      | 0.4421                      | 0.1977                   | (31.1307, 31.9057)                        |
|                                                                        | chrF++     | 17.8943      | 0.4468                      | 0.1998                   | (17.5026, 18.2859)                        |
|                                                                        | COMET      | 0.8066       | 0.0054                      | 0.0024                   | (0.8018, 0.8113)                          |
| **Direct translation without context**                                 | BLEU       | 31.5937      | 0.1074                      | 0.0481                   | (31.4995, 31.6879)                        |
|                                                                        | chrF++     | 18.0681      | 0.1188                      | 0.0531                   | (17.9639, 18.1723)                        |
|                                                                        | COMET      | 0.8292       | 0.0004                      | 0.0002                   | (0.8288, 0.8295)                          |
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 31.4086      | 0.3103                      | 0.1388                   | (31.1365, 31.6806)                        |
|                                                                        | chrF++     | 17.7414      | 0.3499                      | 0.1565                   | (17.4347, 18.0481)                        |
|                                                                        | COMET      | 0.8305       | 0.0022                      | 0.0010                   | (0.8285, 0.8325)                          |


---

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

## Significant testing


| **Context Information**                                               | **Metric** | **Mean (μ)** | **Standard Deviation (σ)** | **Standard Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|-----------------------------|--------------------------|------------------------------------------|
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 33.0993      | 0.3712                      | 0.1660                   | (32.7740, 33.4247)                        |
|                                                                        | chrF++     | 21.0032      | 2.0368                      | 0.9109                   | (19.2179, 22.7886)                        |
|                                                                        | COMET      | 0.8386       | 0.0017                      | 0.0008                   | (0.8370, 0.8401)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 34.7799      | 1.0541                      | 0.4714                   | (33.8559, 35.7038)                        |
|                                                                        | chrF++     | 21.3302      | 0.9015                      | 0.4032                   | (20.5400, 22.1204)                        |
|                                                                        | COMET      | 0.8531       | 0.0014                      | 0.0006                   | (0.8519, 0.8544)                          |
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 32.1151      | 1.1747                      | 0.5254                   | (31.0854, 33.1448)                        |
|                                                                        | chrF++     | 20.8285      | 1.3308                      | 0.5952                   | (19.6620, 21.9950)                        |
|                                                                        | COMET      | 0.8449       | 0.0031                      | 0.0014                   | (0.8422, 0.8476)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 32.9844      | 0.8037                      | 0.3594                   | (32.2800, 33.6889)                        |
|                                                                        | chrF++     | 21.4367      | 0.9011                      | 0.4030                   | (20.6469, 22.2266)                        |
|                                                                        | COMET      | 0.8455       | 0.0024                      | 0.0011                   | (0.8434, 0.8476)                          |
| **Direct translation without context**                                 | BLEU       | 33.0801      | 1.0192                      | 0.4558                   | (32.1867, 33.9735)                        |
|                                                                        | chrF++     | 21.6181      | 0.4784                      | 0.2140                   | (21.1988, 22.0375)                        |
|                                                                        | COMET      | 0.8461       | 0.0015                      | 0.0007                   | (0.8448, 0.8475)                          |
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 33.5307      | 1.2281                      | 0.5492                   | (32.4542, 34.6072)                        |
|                                                                        | chrF++     | 21.9507      | 0.5198                      | 0.2325                   | (21.4951, 22.4063)                        |
|                                                                        | COMET      | 0.8509       | 0.0024                      | 0.0011                   | (0.8489, 0.8530)                          |
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 32.7756      | 1.3068                      | 0.5844                   | (31.6301, 33.9211)                        |
|                                                                        | chrF++     | 20.2092      | 0.9631                      | 0.4307                   | (19.3651, 21.0534)                        |
|                                                                        | COMET      | 0.8387       | 0.0024                      | 0.0011                   | (0.8366, 0.8408)                          |


---

## Meta-Llama-3.1-8B-Instruct (without finetune) (Alpaca)  

| Contextual Information | Direct Avg BLEU | Direct Avg CHRF+ | Direct Avg COMET | Total Data |
|------------------------|----------------|------------------|------------------|------------|
| ✅ Synonyms of each concept derived from GPT-4o Mini | 30.224 | 20.07 | 0.798 | 100 |
| Translation dictionary based on UMLS | 27.54 | 19.23 | 0.7921 | 100 |
| Knowledge graphs of each concept generated using GPT-4o Mini | 25.99 | 15.93 | 0.7458 | 100 |
| Conceptual relationships extracted from the UMLS | 27.03 | 18.16 | 0.7760 | 100 |
| Multilingual translations of each concept obtained from GPT-4o Mini | 28.637 | 21.84 | 0.8083 | 100 |
| Synonyms of each concept obtained from UMLS | 29.37 | 19.69 | 0.7949 | 100 |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context | 27.65 | 14.98 | 0.797 | 100 |

## Significant testing

| **Context Information**                                               | **Metric** | **Mean (μ)** | **Standard Deviation (σ)** | **Standard Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|-----------------------------|--------------------------|------------------------------------------|
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 24.2976      | 0.9176                      | 0.4104                   | (23.4933, 25.1020)                        |
|                                                                        | chrF++     | 16.7070      | 0.6765                      | 0.3026                   | (16.1140, 17.3000)                        |
|                                                                        | COMET      | 0.7513       | 0.0138                      | 0.0062                   | (0.7392, 0.7634)                          |
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 28.2482      | 1.2313                      | 0.5506                   | (27.1690, 29.3275)                        |
|                                                                        | chrF++     | 18.2865      | 1.0307                      | 0.4609                   | (17.3831, 19.1899)                        |
|                                                                        | COMET      | 0.7822       | 0.0078                      | 0.0035                   | (0.7753, 0.7891)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 27.9658      | 1.2026                      | 0.5378                   | (26.9117, 29.0200)                        |
|                                                                        | chrF++     | 20.9455      | 1.4877                      | 0.6653                   | (19.6415, 22.2495)                        |
|                                                                        | COMET      | 0.7874       | 0.0049                      | 0.0022                   | (0.7831, 0.7917)                          |
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 26.9388      | 1.2068                      | 0.5397                   | (25.8810, 27.9966)                        |
|                                                                        | chrF++     | 17.6176      | 0.4743                      | 0.2121                   | (17.2019, 18.0333)                        |
|                                                                        | COMET      | 0.7800       | 0.0059                      | 0.0026                   | (0.7748, 0.7852)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 29.1671      | 0.7036                      | 0.3146                   | (28.5504, 29.7838)                        |
|                                                                        | chrF++     | 19.3669      | 1.4175                      | 0.6339                   | (18.1244, 20.6094)                        |
|                                                                        | COMET      | 0.7991       | 0.0076                      | 0.0034                   | (0.7924, 0.8057)                          |
| **Direct translation without context**                                 | BLEU       | 27.2813      | 1.1677                      | 0.5222                   | (26.2577, 28.3048)                        |
|                                                                        | chrF++     | 15.1596      | 1.3664                      | 0.6111                   | (13.9619, 16.3573)                        |
|                                                                        | COMET      | 0.7878       | 0.0131                      | 0.0059                   | (0.7762, 0.7993)                          |
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 28.2573      | 1.1323                      | 0.5064                   | (27.2648, 29.2498)                        |
|                                                                        | chrF++     | 19.5345      | 0.9229                      | 0.4127                   | (18.7256, 20.3434)                        |
|                                                                        | COMET      | 0.7734       | 0.0118                      | 0.0053                   | (0.7630, 0.7838)                          |

---


---

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



| **Context Information**                                               | **Metric** | **Mean (μ)** | **Standard Deviation (σ)** | **Standard Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|-----------------------------|--------------------------|------------------------------------------|
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 38.0820      | 0.1990                      | 0.0890                   | (37.9076, 38.2564)                        |
|                                                                        | chrF++     | 23.9127      | 0.2534                      | 0.1133                   | (23.6906, 24.1349)                        |
|                                                                        | COMET      | 0.8452       | 0.0006                      | 0.0003                   | (0.8447, 0.8457)                          |
| **Direct translation without context**                                 | BLEU       | 37.7127      | 0.0440                      | 0.0197                   | (37.6741, 37.7512)                        |
|                                                                        | chrF++     | 15.9806      | 0.0819                      | 0.0366                   | (15.9088, 16.0524)                        |
|                                                                        | COMET      | 0.8521       | 0.0005                      | 0.0002                   | (0.8517, 0.8525)                          |
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 38.6849      | 0.1742                      | 0.0779                   | (38.5322, 38.8375)                        |
|                                                                        | chrF++     | 24.2416      | 0.2642                      | 0.1182                   | (24.0100, 24.4732)                        |
|                                                                        | COMET      | 0.8522       | 0.0002                      | 0.0001                   | (0.8520, 0.8525)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 38.7952      | 0.1401                      | 0.0627                   | (38.6724, 38.9181)                        |
|                                                                        | chrF++     | 23.1822      | 0.2766                      | 0.1237                   | (22.9398, 23.4247)                        |
|                                                                        | COMET      | 0.8570       | 0.0002                      | 0.0001                   | (0.8568, 0.8573)                          |
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 39.8393      | 0.2105                      | 0.0941                   | (39.6548, 40.0238)                        |
|                                                                        | chrF++     | 24.0348      | 0.2542                      | 0.1137                   | (23.8119, 24.2576)                        |
|                                                                        | COMET      | 0.8489       | 0.0007                      | 0.0003                   | (0.8483, 0.8495)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 38.5423      | 0.2461                      | 0.1101                   | (38.3266, 38.7580)                        |
|                                                                        | chrF++     | 23.1491      | 0.2233                      | 0.0999                   | (22.9533, 23.3448)                        |
|                                                                        | COMET      | 0.8532       | 0.0008                      | 0.0004                   | (0.8525, 0.8539)                          |
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 38.6538      | 0.2572                      | 0.1150                   | (38.4283, 38.8792)                        |
|                                                                        | chrF++     | 23.7857      | 0.0538                      | 0.0241                   | (23.7385, 23.8329)                        |
|                                                                        | COMET      | 0.8573       | 0.0004                      | 0.0002                   | (0.8570, 0.8576)                          |



---


## Qwen2.5 7B (without finetune) (gpt)  

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|------------------|------------|
| Conceptual relationships extracted from the UMLS         | 25.30           | 16.59            | 0.7598           | 100        |
| Knowledge graphs of each concept generated using GPT-4o Mini | 26.85           | 18.01            | 0.7717           | 100        |
| Synonyms of each concept obtained from UMLS               | 19.29           | 10.88            | 0.6970           | 95         |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 33.40           | 20.35            | 0.8451           | 100        |
| Synonyms of each concept derived from GPT-4o Mini        | 31.07           | 19.14            | 0.8211           | 99         |
| Translation dictionary based on UMLS                      | 24.50           | 16.58            | 0.7607           | 99         |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                        | 26.91           | 20.29            | 0.80           | 100        |


| **Context Information**                                               | **Metric** | **Mean (μ)** | **Std Dev (σ)** | **Std Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|------------------|--------------------|------------------------------------------|
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 25.2761      | 0.0390           | 0.0174             | (25.2419, 25.3103)                        |
|                                                                        | CHRF++     | 16.6243      | 0.0381           | 0.0170             | (16.5908, 16.6577)                        |
|                                                                        | COMET      | 0.7599       | 0.0001           | 0.0001             | (0.7598, 0.7600)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 33.3799      | 0.0005           | 0.0002             | (33.3795, 33.3803)                        |
|                                                                        | CHRF++     | 20.3273      | 0.0008           | 0.0003             | (20.3266, 20.3279)                        |
|                                                                        | COMET      | 0.8450       | 0.0001           | 0.0000             | (0.8449, 0.8451)                          |
| **Direct translation without context**                                 | BLEU       | 31.8525      | 0.0000           | 0.0000             | (31.8525, 31.8525)                        |
|                                                                        | CHRF++     | 18.8863      | 0.0000           | 0.0000             | (18.8863, 18.8863)                        |
|                                                                        | COMET      | 0.8413       | 0.0000           | 0.0000             | (0.8413, 0.8413)                          |
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 26.8483      | 0.0012           | 0.0005             | (26.8472, 26.8493)                        |
|                                                                        | CHRF++     | 17.9346      | 0.0850           | 0.0380             | (17.8601, 18.0091)                        |
|                                                                        | COMET      | 0.7716       | 0.0001           | 0.0000             | (0.7716, 0.7717)                          |
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 31.1615      | 0.0507           | 0.0227             | (31.1171, 31.2059)                        |
|                                                                        | CHRF++     | 19.1172      | 0.0056           | 0.0025             | (19.1122, 19.1221)                        |
|                                                                        | COMET      | 0.8222       | 0.0004           | 0.0002             | (0.8218, 0.8225)                          |
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 19.3368      | 0.0119           | 0.0053             | (19.3263, 19.3472)                        |
|                                                                        | CHRF++     | 10.7996      | 0.0530           | 0.0237             | (10.7532, 10.8460)                        |
|                                                                        | COMET      | 0.6903       | 0.0001           | 0.0001             | (0.6902, 0.6904)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 24.4723      | 0.0371           | 0.0166             | (24.4398, 24.5048)                        |
|                                                                        | CHRF++     | 16.5740      | 0.0252           | 0.0113             | (16.5519, 16.5962)                        |
|                                                                        | COMET      | 0.7575       | 0.0002           | 0.0001             | (0.7573, 0.7576)                          |


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



| **Context Information**                                               | **Metric** | **Mean (μ)** | **Std Dev (σ)** | **Std Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|------------------|--------------------|------------------------------------------|
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 32.0735      | 0.0981           | 0.0439             | (31.9875, 32.1595)                        |
|                                                                        | CHRF++     | 21.0239      | 0.1065           | 0.0476             | (20.9306, 21.1173)                        |
|                                                                        | COMET      | 0.8464       | 0.0001           | 0.0001             | (0.8463, 0.8466)                          |
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 30.0770      | 0.0562           | 0.0251             | (30.0278, 30.1263)                        |
|                                                                        | CHRF++     | 21.7895      | 0.0429           | 0.0192             | (21.7519, 21.8271)                        |
|                                                                        | COMET      | 0.8359       | 0.0001           | 0.0001             | (0.8358, 0.8361)                          |
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 31.2966      | 0.0055           | 0.0025             | (31.2918, 31.3014)                        |
|                                                                        | CHRF++     | 20.5050      | 0.0079           | 0.0035             | (20.4980, 20.5119)                        |
|                                                                        | COMET      | 0.8381       | 0.0000           | 0.0000             | (0.8381, 0.8381)                          |
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 31.7414      | 0.0065           | 0.0029             | (31.7357, 31.7471)                        |
|                                                                        | CHRF++     | 20.6162      | 0.0187           | 0.0083             | (20.5999, 20.6326)                        |
|                                                                        | COMET      | 0.8350       | 0.0001           | 0.0000             | (0.8349, 0.8350)                          |
| **Direct translation without context**                                 | BLEU       | 31.9709      | 0.0357           | 0.0160             | (31.9396, 32.0022)                        |
|                                                                        | CHRF++     | 21.8419      | 0.0306           | 0.0137             | (21.8151, 21.8687)                        |
|                                                                        | COMET      | 0.8372       | 0.0001           | 0.0000             | (0.8371, 0.8373)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 33.6826      | 0.0309           | 0.0138             | (33.6555, 33.7097)                        |
|                                                                        | CHRF++     | 21.5874      | 0.0931           | 0.0416             | (21.5058, 21.6690)                        |
|                                                                        | COMET      | 0.8500       | 0.0002           | 0.0001             | (0.8498, 0.8502)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 32.5127      | 0.0099           | 0.0044             | (32.5041, 32.5214)                        |
|                                                                        | CHRF++     | 20.3291      | 0.0077           | 0.0035             | (20.3223, 20.3359)                        |
|                                                                        | COMET      | 0.8371       | 0.0001           | 0.0001             | (0.8370, 0.8373)                          |


---

## Qwen2.5 3B (without finetune) 

| Contextual Information                                      | Direct Avg BLEU | Direct Avg chrF+ | Direct Avg COMET | Total Data |
|------------------------------------------------------------|-----------------|------------------|------------------|------------|
| Synonyms of each concept obtained from UMLS               | 21.73           | 12.69            | 0.7416           | 98         |
| Translation dictionary based on UMLS                      | 22.73           | 13.52            | 0.7485           | 99         |
| Synonyms of each concept derived from GPT-4o Mini        | 28.31           | 19.44            | 0.8225           | 99         |
| Conceptual relationships extracted from the UMLS         | 23.41           | 13.51            | 0.7679           | 99         |
| Knowledge graphs of each concept generated using GPT-4o Mini | 24.14           | 12.07            | 0.7973           | 98         |
| ✅ Multilingual translations of each concept obtained from GPT-4o Mini | 29.73           | 17.30            | 0.838           | 99         |
| ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) Direct translation without context                        | 19.97           | 13.86            | 0.73           | 100        |

---

| **Context Information**                                               | **Metric** | **Mean (μ)** | **Std Dev (σ)** | **Std Error (SE)** | **95% Confidence Interval (CI)**         |
|------------------------------------------------------------------------|------------|--------------|------------------|--------------------|------------------------------------------|
| **Synonyms of each concept derived from GPT-4o Mini**                  | BLEU       | 27.8040      | 0.0295           | 0.0132             | (27.7781, 27.8299)                        |
|                                                                        | CHRF++     | 18.6636      | 0.0273           | 0.0122             | (18.6397, 18.6875)                        |
|                                                                        | COMET      | 0.8153       | 0.0001           | 0.0001             | (0.8152, 0.8155)                          |
| **Conceptual relationships extracted from the UMLS**                   | BLEU       | 24.2942      | 0.0528           | 0.0236             | (24.2479, 24.3404)                        |
|                                                                        | CHRF++     | 13.1354      | 0.0074           | 0.0033             | (13.1289, 13.1419)                        |
|                                                                        | COMET      | 0.7761       | 0.0020           | 0.0009             | (0.7744, 0.7778)                          |
| **Translation dictionary based on UMLS**                               | BLEU       | 22.7263      | 0.0848           | 0.0379             | (22.6520, 22.8007)                        |
|                                                                        | CHRF++     | 14.2825      | 0.0925           | 0.0414             | (14.2014, 14.3636)                        |
|                                                                        | COMET      | 0.7636       | 0.0007           | 0.0003             | (0.7629, 0.7642)                          |
| **Direct translation without context**                                 | BLEU       | 28.5633      | 0.1149           | 0.0514             | (28.4625, 28.6640)                        |
|                                                                        | CHRF++     | 17.2538      | 0.0495           | 0.0221             | (17.2104, 17.2972)                        |
|                                                                        | COMET      | 0.8272       | 0.0021           | 0.0010             | (0.8253, 0.8291)                          |
| **Multilingual translations of each concept obtained from GPT-4o Mini**| BLEU       | 28.0129      | 0.0651           | 0.0291             | (27.9558, 28.0700)                        |
|                                                                        | CHRF++     | 18.6306      | 0.0005           | 0.0002             | (18.6302, 18.6311)                        |
|                                                                        | COMET      | 0.8182       | 0.0001           | 0.0000             | (0.8182, 0.8183)                          |
| **Synonyms of each concept obtained from UMLS**                        | BLEU       | 21.2897      | 0.0247           | 0.0110             | (21.2680, 21.3113)                        |
|                                                                        | CHRF++     | 12.5402      | 0.0189           | 0.0085             | (12.5236, 12.5568)                        |
|                                                                        | COMET      | 0.7414       | 0.0001           | 0.0001             | (0.7413, 0.7415)                          |
| **Knowledge graphs of each concept generated using GPT-4o Mini**       | BLEU       | 24.0598      | 0.0726           | 0.0325             | (23.9961, 24.1234)                        |
|                                                                        | CHRF++     | 12.1175      | 0.0331           | 0.0148             | (12.0885, 12.1464)                        |
|                                                                        | COMET      | 0.7916       | 0.0001           | 0.0000             | (0.7916, 0.7917)                          |

---



## gemma-3-4b-it (finetune) 


| Contextual Information                                      | Direct Avg BLEU | Direct Avg CHRF++ | Direct Avg COMET | Total Number of Data |
|-------------------------------------------------------------|------------------|--------------------|-------------------|-----------------------|
| Multilingual translations of each concept obtained from GPT-4o Mini | 33.54           | 20.65             | 0.8504            | 92                    |
| Knowledge graphs of each concept generated using GPT-4o Mini | 31.03           | 20.54             | 0.8042            | 100                   |
| Translation dictionary based on UMLS                         | 31.94           | 21.00             | 0.8031            | 100                   |
| Direct translation without context                           | 31.08           | 20.36             | 0.8084            | 100                   |
| Conceptual relationships extracted from the UMLS             | 31.50           | 20.16             | 0.8046            | 100                   |
| Synonyms of each concept derived from GPT-4o Mini            | 35.22           | 19.83             | 0.8507            | 99                    |
| Synonyms of each concept obtained from UMLS                  | 32.42           | 21.73             | 0.8054            | 100                   |


---

## gemma-3-4b-it (without finetune) 

| Contextual Information                              | Direct Avg BLEU | Direct Avg CHRF+ | Direct Avg COMET | Total Number of Data |
|-----------------------------------------------------|-----------------|------------------|------------------|----------------------|
| Synonyms of each concept obtained from UMLS         | 32.359722       | 21.726360        | 0.805463         | 100                  |
| Translation dictionary based on UMLS                | 31.842232       | 21.301213        | 0.803096         | 100                  |
| Direct translation without context                  | 31.166060       | 20.349546        | 0.808483         | 100                  |
| Multilingual translations from GPT-4o Mini          | 31.722875       | 21.462878        | 0.801598         | 100                  |
| Knowledge graphs from GPT-4o Mini                   | 31.150204       | 20.547009        | 0.804385         | 100                  |
| Conceptual relationships extracted from UMLS        | 31.636596       | 19.933975        | 0.804431         | 100                  |
| Synonyms of each concept derived from GPT-4o Mini   | 32.279868       | 20.902552        | 0.807744         | 100                  |

---



---
** (Finetune + With Context) > (Finetune + No Context) > (Without Finetune + With Context) > (Without Finetune + No Context) **

## 1) (Without Finetune + With Context) vs (Without Finetune + No Context)
## 2) (Finetune + No context) vs. (Without Finetune + No Context) 
## 3) (Finetune + With Context) vs (Finetune + No Context)

---

### 1) **Without Finetune + With Context** vs **Without Finetune + No Context**


| Model                              | Context Type                    | BLEU ↑  | chrF+ ↑ | COMET ↑  |
|-----------------------------------|----------------------------------|--------|--------|----------|
| **Phi-4**                         | ✅ Translation Dictionary        | 35.89  | 20.17  | 0.835    |
|                                   | ❌ No Context                    | 18.77  | 8.79   | 0.643    |
| **Qwen2.5 14B**                   | ✅ Multilingual Translations     | 41.95  | 25.93  | 0.8614   |
|                                   | ❌ No Context                    | 38.63  | 23.53  | 0.8491   |
| **Meta-Llama-3.1-8B**             | ✅ GPT-4o Synonyms               | 30.22  | 20.07  | 0.798    |
|                                   | ❌ No Context                    | 29.36  | 18.94  | 0.8113   |
| **Qwen2.5 7B**                    | ✅ GPT-4o Multilingual           | 33.40  | 20.35  | 0.8451   |
|                                   | ❌ No Context                    | 31.77  | 21.54  | 0.8402   |
| **Qwen2.5 3B**                    | ✅ GPT-4o Multilingual           | 29.73  | 17.30  | 0.838    |
|                                   | ❌ No Context                    | 28.62  | 17.19  | 0.8274   |
| **Gemma-3-4B-it**                 | ✅ UMLS Synonyms                 | 32.36  | 21.73  | 0.8055   |
|                                   | ❌ No Context                    | 31.17  | 20.35  | 0.8085   |
---



### 2) **Direct Translation Without Context — Finetuned vs. Without Finetune**



| Model                              | Context Type                    | BLEU ↑  | chrF+ ↑ | COMET ↑   |
|-----------------------------------|----------------------------------|--------|--------|-----------|
| **Phi-4**                         | ✅ Finetune                      | 42.52  | 28.35  | 0.86159   |
|                                   | ❌ No Finetune                  | 18.77  | 8.79   | 0.643     |
| **Qwen2.5 14B**                   | ✅ Finetune                      | 38.63  | 23.53  | 0.8491    |
|                                   | ❌ No Finetune                  | 34.13  | 21.28  | 0.8314    |
| **Qwen2.5 7B**                    | ✅ Finetune                      | 39.09  | 23.94  | 0.8555    |
|                                   | ❌ No Finetune                  | 31.77  | 21.54  | 0.8402    |
| **Qwen2.5 3B**                    | ✅ Finetune                      | 38.83  | 22.33  | 0.856     |
|                                   | ❌ No Finetune                  | 28.62  | 17.19  | 0.8274    |
| **Meta-Llama-3.1-8B**             | ✅ Finetune                      | 34.47  | 22.11  | 0.8502    |
|                                   | ❌ No Finetune                  | 29.36  | 18.94  | 0.8113    |
| **Gemma-3-4B-it**                 | ✅ Finetune                      | 31.08  | 20.36  | 0.8084    |
|                                   | ❌ No Finetune                  | 31.17  | 20.35  | 0.8085    |                                 | ❌ No Finetune                  | 29.36  | 18.94  | 0.8113    |


---

### 3) **(Finetune + With Context)** vs **(Finetune + No Context)**


| Model                                | Context Type        | Avg BLEU | Avg chrF+ | Avg COMET |
|-------------------------------------|---------------------|----------|-----------|-----------|
| **Phi-4**                            | ✅ Multilingual (context) | **44.23** | 28.91     | **0.86299** |
|                                     | ❌ No context              | 42.52    | 28.35     | 0.86159   |
| **Qwen2.5 14B**                      | ✅ Multilingual (context) | **41.95** | 25.93     | **0.8614**  |
|                                     | ❌ No context              | 38.63    | 23.53     | 0.8491    |
| **Meta-Llama-3.1-8B-Instruct**       | ✅ Synonyms GPT-4o (context) | **33.12** | 23.25     | **0.8526**  |
|                                     | ❌ No context              | 34.47    | 22.11     | 0.8502    |
| **Qwen2.5 7B**                       | ✅ Knowledge Graphs (context) | **40.47** | **24.58** | 0.8487    |
|                                     | ❌ No context              | 39.09    | 23.94     | **0.8555** |
| **Qwen2.5 3B**                       | ✅ Knowledge Graphs (context) | **40.88** | **24.61** | 0.851     |
|                                     | ❌ No context              | 38.83    | 22.33     | **0.856**  |
| **Gemma-3-4B-it**                    | ✅ Synonyms GPT-4o (context) | **35.33** | 19.96     | **0.8507**  |
|                                     | ❌ No context              | 31.08    | 20.36     | 0.8084    | 

![image](https://github.com/user-attachments/assets/1762d2ec-1de9-451f-a0b0-04b4894517e7)




