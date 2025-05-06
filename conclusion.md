## Case A

### **1. Qualitative and Error Analysis: Fine-Tuning Improves Fluency and Structural Completeness**

Table 5 illustrates how fine-tuning enhances translation accuracy for complex medical instructions. In case A, the base model produces a mostly accurate but disjointed output, splitting the sentence into two independent clauses and omitting the connective rationale present in the original Spanish reference. The fine-tuned version improves cohesion and better mirrors the syntactic structure of the reference by restoring the conjunctive phrase *“y obtener…”* and preserving the causal relationship conveyed in the original. This supports our quantitative findings from Table 2, where fine-tuning leads to statistically significant improvements in BLEU and COMET scores.

---

| Case          | Source Sentence (English)          | Reference (Spanish)                          | Base Output (Direct)                                                      | Improved Output (Finetuned)                                                     | Observation                                            |
| ------------- | ---------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| A. FT vs Base | Veins and arteries vary in size... | Las venas y las arterias varían en tamaño... | Las venas y arterias varían en tamaño... Obtener una muestra de sangre... | Las venas y las arterias varían en tamaño... y obtener una muestra de sangre... | Fine-tuning restores conjunction and improves fluency. |

**Table 5:** Error Analysis Example: Effect of Fine-tuning on Translation Coherence and Completeness

---

## Case B


### **2. Structured Prompting (MedCOD) vs Base: Restoring Clinical Completeness**

Table 6 compares the effect of structured prompting using MedCOD to the base translation in handling long, domain-specific sentences. The MedCOD-enhanced output captures more precise noun phrases like *“las venas y las arterias”*, matching the original reference more closely than the base, which omits articles and flattens nuance. While both outputs retain the core meaning, the MedCOD version also better aligns with the syntactic flow and preserves clause-level structure, improving readability and accuracy. This highlights MedCOD’s value in promoting structural and lexical precision for biomedical content.

---

| Case              | Source Sentence (English)          | Reference (Spanish)                          | Base Output (Direct)                                            | Improved Output (MedCOD)                                            | Observation                                                     |
| ----------------- | ---------------------------------- | -------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------- |
| B. MedCOD vs Base | Veins and arteries vary in size... | Las venas y las arterias varían en tamaño... | Las venas y arterias varían en tamaño... Obtener una muestra... | Las venas y las arterias varían en tamaño... Obtener una muestra... | MedCOD restores noun phrase completeness and lexical precision. |

**Table 6:** Ablation Case Study: Effect of MedCOD Prompting on Biomedical Translation Accuracy

## Case C

Here is the **Case C** qualitative and error analysis in the same style as your original Table 4, highlighting the combined effect of **MedCOD + Fine-tuning**:

---

### **3. Combined MedCOD + Fine-Tuning: Restoring Domain-Specific Clarity and Fluency**

Table 7 highlights how the joint application of MedCOD and fine-tuning enhances both terminological accuracy and syntactic fluency. In this case, while fine-tuning alone improves baseline coverage, it fails to fully normalize clinical phrasing, misrepresents plural agreement (*“transpuesto”*), and misses article use (*“La transposición”*). The MedCOD + FT output corrects all of these: it reinstates the article, preserves accurate noun phrase structure (*“los grandes vasos”*), and correctly pluralizes the translated medical term (*“transpuestos”*). This demonstrates how combining structured context with tuning enables domain-specific fluency that aligns closely with expert biomedical Spanish.

---

| Case               | Source Sentence (English)                               | Reference (Spanish)                                      | Base Output (FT only)                                                             | Improved Output (MedCOD+FT)                                                           | Observation                                                    |
| ------------------ | ------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| C. MedCOD+FT vs FT | Transposition of the great vessels is a heart defect... | Es un defecto cardíaco que ocurre desde el nacimiento... | Transposición de los grandes vasos es un defecto... están cambiados (transpuesto) | La transposición de los grandes vasos es un defecto... están cambiados (transpuestos) | MedCOD+FT reinforces grammatical accuracy and domain phrasing. |

**Table 7:** Case Study: Joint Effect of MedCOD and Fine-Tuning on Biomedical Translation Precision

---





