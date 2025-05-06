## Case A

### **1. Qualitative and Error Analysis: Fine-Tuning Improves Fluency and Structural Completeness**

Table 5 illustrates how fine-tuning enhances translation accuracy for complex medical instructions. In this case, the base model produces a mostly accurate but disjointed output, splitting the sentence into two independent clauses and omitting the connective rationale present in the original Spanish reference. The fine-tuned version improves cohesion and better mirrors the syntactic structure of the reference by restoring the conjunctive phrase *“y obtener…”* and preserving the causal relationship conveyed in the original. This supports our quantitative findings from Table 2, where fine-tuning leads to statistically significant improvements in BLEU and COMET scores.

---

| Case          | Source Sentence (English)          | Reference (Spanish)                          | Base Output (Direct)                                                      | Improved Output (Finetuned)                                                     | Observation                                            |
| ------------- | ---------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| A. FT vs Base | Veins and arteries vary in size... | Las venas y las arterias varían en tamaño... | Las venas y arterias varían en tamaño... Obtener una muestra de sangre... | Las venas y las arterias varían en tamaño... y obtener una muestra de sangre... | Fine-tuning restores conjunction and improves fluency. |

**Table 5:** Error Analysis Example: Effect of Fine-tuning on Translation Coherence and Completeness

---


