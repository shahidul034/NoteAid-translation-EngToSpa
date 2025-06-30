# Multiple context mixed
## "src": 
"Some patients develop heart failure and high blood pressure in the blood vessels of their lungs.Your chances of recovering or of the disease getting worse depend on the cause and how severe the disease was when it was first diagnosed."
## "mt":
 "Algunos pacientes desarrollan insuficiencia cardíaca y presión arterial alta en los vasos sanguíneos de sus pulmones. Sus posibilidades de recuperarse o que la enfermedad empeore dependen de la causa y de cuán grave era la enfermedad cuando fue diagnosticada por primera vez."
## "ref": 
"El estrés causado por la enfermedad se puede aliviar a menudo al participar en un grupo de apoyo, en el que los integrantes comparten experiencias y problemas en común."
## "BLEU": 2.4240945815868544,
## "CHRF++": 26.881914623626578,
## "context":
 "Synonyms of 'heart failure in different languages':\nEnglish: [heart failure, cardiac failure, congestive heart failure]; French: [insuffisance cardiaque, défaillance cardiaque]; Portuguese: [insuficiência cardíaca, falência cardíaca]; German: [Herzinsuffizienz, Herzversagen]; Spanish: [insuficiencia cardíaca, fallo cardíaco]\nSynonyms of 'lungs in different languages':\nEnglish: [lungs]; French: [poumons]; Portuguese: [pulmões]; German: [Lungen]; Spanish: [pulmones]\nSynonyms of 'blood vessels in different languages':\nEnglish: [blood vessels, vascular system, veins and arteries]; French: [vaisseaux sanguins, système vasculaire, veines et artères]; Portuguese: [vasos sanguíneos, sistema vascular, veias e artérias]; German: [Blutgefäße, Gefäßsystem, Venen und Arterien]; Spanish: [vasos sanguíneos, sistema vascular, venas y arterias]\nSynonyms of 'recovering in different languages':\nEnglish: [recovering, healing, restoring, regaining, rebounding]; French: [récupération, guérison, restauration, retour, réhabilitation]; Portuguese: [recuperação, restauração, cura, regresso, recuperando]; German: [Erholung, Wiederherstellung, Genesung, Rückgewinnung, Wiederbelebung]; Spanish: [recuperación, sanación, restauración, regreso, recobrando]\nSynonyms of 'diagnosed in different languages':\nEnglish: [identified, determined, recognized]; French: [diagnostiqué, identifié, déterminé]; Portuguese: [diagnosticado, identificado, determinado]; German: [diagnostiziert, erkannt, bestimmt]; Spanish: [diagnosticado, identificado, determinado]\nSynonyms of 'high blood pressure in different languages':\nEnglish: [hypertension, high blood pressure]; French: [hypertension, pression artérielle élevée]; Portuguese: [hipertensão, pressão arterial alta]; German: [Bluthochdruck, Hypertonie]; Spanish: [hipertensión, presión arterial alta]\nSynonyms of 'disease in different languages':\nEnglish: [illness, sickness, disorder, condition]; French: [maladie, affection, pathologie]; Portuguese: [doença, mal, afecção]; German: [Krankheit, Erkrankung, Leiden]; Spanish: [enfermedad, dolencia, trastorno]\n- heart failure: French: \"insuffisance cardiaque\", Portuguese: \"insuficiência cardíaca\", German: \"Herzinsuffizienz\", Spanish: \"insuficiencia cardíaca\"\n- high blood pressure: French: \"hypertension artérielle\", Portuguese: \"hipertensão arterial\", German: \"Bluthochdruck\", Spanish: \"hipertensión arterial\"\n- blood vessels: French: \"vaisseaux sanguins\", Portuguese: \"vasos sanguíneos\", German: \"Blutgefäße\", Spanish: \"vasos sanguíneos\"\n- lungs: French: \"poumons\", Portuguese: \"pulmões\", German: \"Lungen\", Spanish: \"pulmones\"\n- recovering: French: \"récupération\", Portuguese: \"recuperando\", German: \"wiederherstellen\", Spanish: \"recuperando\"\n- disease: French: \"maladie\", Portuguese: \"doença\", German: \"Krankheit\", Spanish: \"enfermedad\"\n- diagnosed: French: \"diagnostiqué\", Portuguese: \"diagnosticado\", German: \"diagnostiziert\", Spanish: \"diagnosticado\"\nChain of dictionary: heart failure means insuficiencia cardíaca means Insuffisances cardiaques means Herzversagen means Insuficiências cardíacas. high blood pressure means presión arterial alta means Hypertension artérielle means Hypertonie means Hipertensão. blood vessels means los vasos sanguíneos means Vaisseaux sanguins means Blutgefäße means Vasos Sanguíneos. lungs means pulmones y means Poumon means Lunge means Pulmão. disease means enfermedad means Maladies sexuellement transmissibles means Sexuell übertragbare Krankheiten means Infecções Sexualmente Transmissíveis. diagnosed means diagnosticado means Souffles cardiaques fonctionnels et non diagnostiqués means funktionelles und nicht-diagnostiziertes Herzgeraeusch means Sopros cardíacos funcionais e não diagnosticados."

---

# Possible answer: 

### ❌ Observation: Context Injection Degrades Translation Quality

In the example provided, the machine translation (MT) output is:

> **MT:** *"Algunos pacientes desarrollan insuficiencia cardíaca y presión arterial alta en los vasos sanguíneos de sus pulmones. Sus posibilidades de recuperarse o que la enfermedad empeore dependen de la causa y de cuán grave era la enfermedad cuando fue diagnosticada por primera vez."*

While this output appears medically coherent and accurate, the evaluation metrics—**BLEU: 2.42** and **chrF++: 26.88**—are **very low**, due to **mismatch with an unrelated reference**:

> **Reference:** *"El estrés causado por la enfermedad se puede aliviar a menudo al participar en un grupo de apoyo..."*

So, the poor metric scores are not necessarily due to poor translation but because of **mismatched reference**, which is a separate issue. However, even putting that aside, the **use of synonym-context did not help improve anything**. Here's why:

---

### 🔬 Logical Explanation: Why Synonym-based Context Harms Performance

#### 1. **Context Overload and Misalignment**

You fed the model long lists of cross-lingual synonyms (e.g., for “heart failure,” “lungs,” “disease,” etc.). This may:

* **Overwhelm the attention mechanism** with redundant lexical items.
* **Distract** the model from focusing on the source sentence.
* Cause **semantic drift**: the model may hallucinate or conflate terms due to the many near-synonyms and language switches.

This violates the principle of *"relevance-focused conditioning"* — context should be **targeted, minimal, and task-specific**, not exhaustive.

#### 2. **Incoherent Context Construction**

The dictionary-style context is not integrated into a narrative or structured format. It's:

* Presented in a list form, lacking syntactic and semantic cohesion.
* Not grounded in the specific sentence being translated.

This makes it hard for the model to learn **how and where** to use the provided terms. Models trained on natural text may **fail to utilize raw lists** effectively.

#### 3. **Lack of Dynamic Context-Target Binding**

The translation model likely doesn’t have a mechanism (like retrieval-augmented generation or keyword-slot binding) to **map synonyms to specific source tokens dynamically**. It’s not instructed to **prefer those synonyms** or map them properly to the original sentence's terms.

Hence, the synonyms are ignored or used incorrectly, leading to:

* No improvement in term selection.
* Potential confusion in generation.

#### 4. **Confounding Multilingual Lexicons**

The inclusion of **multiple languages in the context** (e.g., German, French, Portuguese) introduces **noise** for a model not trained on **multilingual synonym disambiguation** or cross-lingual mapping within inference.

In zero-shot settings or monolingual generation, this multilingual noise may reduce performance or reliability of word choice.

---

### 🧪 Suggested Framing in Your Paper

> Although the hypothesis was that adding domain-specific synonym context would assist the translation model in selecting accurate and medically appropriate terminology, the results show the contrary. When extended lists of multilingual synonyms were injected as context, performance either stagnated or degraded. This appears to result from context overload, lack of sentence-specific alignment, and the absence of semantic grounding. Furthermore, the multilingual synonym lists introduced noise into the attention mechanism, possibly causing confusion during generation.

---

### ✅ Recommendation for Future Work

You can also write:

> Future approaches should explore more targeted context injection, such as **sentence-specific retrieval** of synonyms or using **structured medical ontologies (e.g., UMLS) with disambiguation control**. Fine-tuning on **in-domain translation datasets with soft constraints** on terminology may be more effective than feeding raw synonym lists.

---


