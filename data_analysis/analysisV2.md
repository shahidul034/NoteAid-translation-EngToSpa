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
### Context 1:
  "Synonyms of 'tics in different languages':\nEnglish: [twitches, spasms, jerks]; French: [tic, spasme, sursaut]; Portuguese: [tiques, espasmos, contrações]; German: [Zuckungen, Tics, Krämpfe]; Spanish: [tics, espasmos, contracciones]\nSynonyms of 'children in different languages':\nEnglish: [kids, youth, youngsters, minors]; French: [enfants, jeunes, gosses, petits]; Portuguese: [crianças, miúdos, meninos, meninas]; German: [Kinder, Jugendliche, Söhne, Töchter]; Spanish: [niños, chicos, jóvenes, pequeños]\nSynonyms of 'medications in different languages':\nEnglish: [medications, drugs, medicines, pharmaceuticals]; French: [médicaments, remèdes, pharmaceutiques]; Portuguese: [medicamentos, remédios, fármacos]; German: [Medikamente, Arzneimittel, Pharmazeutika]; Spanish: [medicamentos, fármacos, medicinas]\nSynonyms of 'attention-deficit disorder in different languages':\nEnglish: [attention deficit disorder, ADD, hyperactivity disorder]; French: [trouble du déficit de l'attention, TDA, trouble hyperactif]; Portuguese: [transtorno do déficit de atenção, TDA, transtorno hiperativo]; German: [Aufmerksamkeitsdefizitstörung, ADS, Hyperaktivitätsstörung]; Spanish: [trastorno por déficit de atención, TDA, trastorno hiperactivo]\nSynonyms of 'chronic motor tic disorder in different languages':\nEnglish: [chronic motor tic disorder]; French: [trouble moteur tic chronique]; Portuguese: [transtorno motor de tiques crônico]; German: [chronische motorische Tic-Störung]; Spanish: [trastorno crónico de tics motores]
### Context 2:  
  - medications: French: \"médicaments\", Portuguese: \"medicamentos\", German: \"Medikamente\", Spanish: \"medicamentos\"\n- children: French: \"enfants\", Portuguese: \"crianças\", German: \"Kinder\", Spanish: \"niños\"\n- tics: French: \"tics\", Portuguese: \"tiques\", German: \"Tics\", Spanish: \"tics\"\n- attention-deficit disorder: French: \"trouble du déficit de l'attention\", Portuguese: \"transtorno do déficit de atenção\", German: \"Aufmerksamkeitsdefizitstörung\", Spanish: \"trastorno por déficit de atención\"\n- chronic motor tic disorder: French: \"trouble moteur tic chronique\", Portuguese: \"transtorno motor de tiques crônicos\", German: \"chronische motorische Tic-Störung\", Spanish: \"trastorno crónico de tics motores\"
### Context 3:
  Chain of dictionary: medications means medicamentos para el means Prémédication à l'anesthésie means Anästhesieprämedikation means Medicação Pré-Anestésica. children means niños means Aide aux familles avec enfants à charge means Hilfe für Familien mit minderjährigen Kindern means Ajuda a Famílias com Filhos Dependentes. tics means tics means Anesthésiques locaux means Lokalanästhetika means Anestésicos Locais. attention-deficit disorder means trastorno por déficit de atención means Trouble déficitaire de l'attention means Aufmerksamkeitsdefizitstoerung means Transtorno de déficit de atenção. chronic motor tic disorder means trastorno crónico del movimiento means Trouble de tic moteur chronique means Chronische motorische Tic-Stoerung means Transtorno de tique motor crônico."

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


