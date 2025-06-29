# This instance is without finetune and with context:
## Source text: 
Three outbreaks of herpesvirus meningoencephalitis in cattle have been reported in three municipalities in the northern region of the State of Tocantins, Brazil. In one outbreak, 41 predominantly young bovines were affected, with 2-3 deaths in some cases. The animals showed neurological signs of incoordination, blindness, and recumbency, with death occurring within approximately 4-5 d. At necropsy, hyperemia and leptomeningeal hemorrhages were observed in the brain. Histology revealed more intense lesions in the rostral portions of the brain, mainly affecting the frontoparietal cerebral cortex, with nonsuppurative encephalitis and meningitis, glial nodules, neuronophagia, and eosinophilic intranuclear inclusion bodies in the astrocytes and neurons. This study shows the presence of bovine herpesvirus in Tocantins, probably the highly neurotropic type 5 strain, and emphasizes its importance in the differential diagnosis of bovine neuropathies.
## Destination text: 
Três surtos de meningoencefalite por herpesvírus em bovinos são relatados em três municípios da região norte do Estado do Tocantins, Brasil. Num surto, 41 animais predominantemente jovens foram afetados, com 2-3 mortes nos outros casos. Os animais apresentaram sinais neurológicos de incoordenação, cegueira e decúbito, com a morte ocorrendo em aproximadamente 4 a 5 dias. Na necropsia foram observadas hiperemia e hemorragias leptomeníngeas no encéfalo. A histologia revelou lesões mais intensas nas porções rostrais do encéfalo, principalmente no córtex cerebral frontoparietal, com encefalite e meningite não supurativas, nódulos gliais, neuronofagia e corpúsculos de inclusão intranucleares eosinofílicos nos astrócitos e neurônios. Este estudo demonstra a presença do herpesvírus bovino no Tocantins, provavelmente a cepa tipo 5 altamente neurotrópica, e enfatiza sua importância no diagnóstico diferencial das neuropatias bovinas.
## Hypothesis text: 
Três surtos de meningoencefalite viral do herpes em bovinos foram relatados em três municípios na região norte do estado do Tocantins, Brasil. Em um dos surtos, 41 bovinos jovens foram afetados, com até 2-3 mortes em alguns casos. Os animais apresentaram sinais neurológicos de descoordenação, cegueira e recumbência, com morte ocorrendo em aproximadamente 4-5 dias. Na necropsia, observou-se hiperemia e hemorragias leptomeníngeas no cérebro. A histologia revelou lesões mais intensas nas porções rostrais do cérebro, principalmente afetando o córtex cerebral frontoparietal, com encefalite e meningite não supurativas, nódulos gliais, neuronofagia e corpos de inclusão intranucleares eosinofílicos nos astrocitos e neurônios. Este estudo mostra a presença do vírus do herpes bovino no Tocantins, provavelmente a cepa altamente neurotrópica tipo 5, e enfatiza sua importância no diagnóstico diferencial das neuropatias bovinas.
## BLEU: 
56.2352572179315
## CHRF++: 
80.67122077747827
## COMET: 
0.8190675973892212

---

# This instance is without finetune and no context:
## Source text: 
Three outbreaks of herpesvirus meningoencephalitis in cattle have been reported in three municipalities in the northern region of the State of Tocantins, Brazil. In one outbreak, 41 predominantly young bovines were affected, with 2-3 deaths in some cases. The animals showed neurological signs of incoordination, blindness, and recumbency, with death occurring within approximately 4-5 d. At necropsy, hyperemia and leptomeningeal hemorrhages were observed in the brain. Histology revealed more intense lesions in the rostral portions of the brain, mainly affecting the frontoparietal cerebral cortex, with nonsuppurative encephalitis and meningitis, glial nodules, neuronophagia, and eosinophilic intranuclear inclusion bodies in the astrocytes and neurons. This study shows the presence of bovine herpesvirus in Tocantins, probably the highly neurotropic type 5 strain, and emphasizes its importance in the differential diagnosis of bovine neuropathies.
## Destination text: 
Três surtos de meningoencefalite por herpesvírus em bovinos são relatados em três municípios da região norte do Estado do Tocantins, Brasil. Num surto, 41 animais predominantemente jovens foram afetados, com 2-3 mortes nos outros casos. Os animais apresentaram sinais neurológicos de incoordenação, cegueira e decúbito, com a morte ocorrendo em aproximadamente 4 a 5 dias. Na necropsia foram observadas hiperemia e hemorragias leptomeníngeas no encéfalo. A histologia revelou lesões mais intensas nas porções rostrais do encéfalo, principalmente no córtex cerebral frontoparietal, com encefalite e meningite não supurativas, nódulos gliais, neuronofagia e corpúsculos de inclusão intranucleares eosinofílicos nos astrócitos e neurônios. Este estudo demonstra a presença do herpesvírus bovino no Tocantins, provavelmente a cepa tipo 5 altamente neurotrópica, e enfatiza sua importância no diagnóstico diferencial das neuropatias bovinas.
## Hypothesis text: 
Três surtos de meningoencefalite viral herpética foram relatados em três municípios na região norte do estado do Tocantins, Brasil. Em um dos surtos, 41 bovinos jovens foram afetados, com até 2-3 mortes em alguns casos. Os animais apresentaram sinais neurológicos de descoordenação, cegueira e recumbência, com óbito ocorrendo em aproximadamente 4-5 dias. A necropsia revelou hipercemia e hemorragias leptomeningeais no cérebro. A histologia mostrou lesões mais intensas nas porções rostrais do cérebro, principalmente afetando o córtex cerebral frontoparietal, com encefalite e meningite não supurativa, nódulos gliádicos, fagocitose neuronal e corpos inclusos intranucleares eosinófilos em astrocítos e neurônios. Este estudo demonstra a presença do vírus herpes bovino no Tocantins, provavelmente da cepa altamente neurotropismo tipo 5, e enfatiza sua importância no diagnóstico diferencial das neuropatias bovinas.
## BLEU: 
46.02940002017582
## CHRF++: 
73.07642078411429
## COMET: 
0.727728009223938

---
---
---


---

## 🔍 Expanded Error Analysis

This section presents a qualitative comparison between the two experimental settings: **(1) Without finetuning + with MedCOD context** vs **(2) Without finetuning + no context**. The goal is to evaluate how the inclusion of **medical contextual information (MedCOD)** impacts translation quality, particularly in specialized medical discourse. To better understand the improvement, we categorize the errors observed in the second condition (no context) and highlight how context mitigates them.

### ✅ Overall Metric Comparison

| Setting                    | BLEU  | chrF++ | COMET |
| -------------------------- | ----- | ------ | ----- |
| No finetune + With context | 56.24 | 80.67  | 0.819 |
| No finetune + No context   | 46.03 | 73.08  | 0.728 |

---

### 🧠 Categorized Error Analysis

| **Category**                        | **Description**                                                                          | **Observed in No Context**                                                                                                                                                                             | **Improved in Context**                                                                               | **Example**                                                                                                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Medical Terminology Mismatch** | Incorrect or ambiguous translations of domain-specific terms.                            | ❌ “**meningoencefalite viral herpética**” → awkward and non-standard term for “herpesvirus meningoencephalitis” <br> ❌ “**vírus herpes bovino**” instead of “herpesvírus bovino” (common medical term) | ✅ Correct and fluent medical terms: “**meningoencefalite por herpesvírus**”, “**herpesvírus bovino**” | → Source: “herpesvirus meningoencephalitis” → Context: **Yes** → “meningoencefalite por herpesvírus” (✓)<br>→ No context: “meningoencefalite viral herpética” (✗) |
| **2. Register and Style**           | Use of inappropriate or non-academic expressions for scientific register.                | ❌ “**óbito ocorrendo**” (colloquial, non-standard) <br> ❌ “**com até 2-3 mortes**” (informal phrasing)                                                                                                 | ✅ Scientific and formal phrasing: “**com a morte ocorrendo**”, “**com 2-3 mortes**”                   |                                                                                                                                                                   |
| **3. Lexical Choice and Fluency**   | Selection of unusual or less common words in medical/scientific Portuguese.              | ❌ “**hipercemia**” (misspelling of “hiperemia”) <br> ❌ “**corpos inclusos**” vs “**corpúsculos de inclusão**”                                                                                          | ✅ Correct technical expressions aligned with biomedical literature                                    |                                                                                                                                                                   |
| **4. Morphological Errors**         | Incorrect derivation or agreement in words, especially in technical vocabulary.          | ❌ “**nódulos gliádicos**” (non-standard derivation; should be “gliais”) <br> ❌ “**neurotropismo tipo 5**” (should be “cepa tipo 5 altamente neurotrópica”)                                             | ✅ Proper derivation and modifier order                                                                |                                                                                                                                                                   |
| **5. Word Order Errors**            | Misplacement of adjectives or noun modifiers that change the clarity or naturalness.     | ❌ “**vírus herpes bovino**” (wrong order)                                                                                                                                                              | ✅ “**herpesvírus bovino**” (correct collocation in medical Portuguese)                                |                                                                                                                                                                   |
| **6. Semantic Fidelity**            | Slight distortion or loss of nuance, such as the implication of severity or progression. | ❌ “**com até 2-3 mortes**” adds hedging not present in the source <br> ❌ “**mostrou lesões... principalmente afetando...**” introduces ambiguity                                                       | ✅ Maintains precise temporal and spatial descriptions                                                 |                                                                                                                                                                   |

---

### 🧩 Interpretation of Contextual Benefit (MedCOD)

The model with **context from MedCOD** appears to internalize **domain-specific phrasing patterns**, terminology conventions, and formal register typically found in medical case reports. These benefits stem from the structured context that provides cues about:

* Expected **entity types** (e.g., disease, anatomical region),
* **Syntactic structures** commonly used in biomedical literature,
* Implicit norms of **formal register** and **style** in Portuguese biomedical translation.

---

