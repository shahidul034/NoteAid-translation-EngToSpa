

# Enhancing English-to-Spanish Medical Translation of Large Language Models Using Enriched Chain-of-Dictionary Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Methods](#methods)  
  - [Overview](#overview)  
  - [Data Source](#data-source)  
  - [MedCOD Framework](#medcod-framework)  
  - [Fine-tuning with LoRA](#fine-tuning-with-lora)  
  - [Experimental Setup](#experimental-setup)  
- [Results](#results)  
- [Ablation Study](#ablation-study)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)  
- [Contact](#contact)  

---

## Project Overview

This project evaluates open-source large language models (LLMs) for English-to-Spanish medical text translation, introducing **MedCOD** — a novel framework that enriches LLM prompts with structured medical knowledge from UMLS and LLM-KB (a large language model knowledge base). We explore different prompting strategies and fine-tuning techniques to improve translation accuracy and robustness in clinical contexts.

---

## Methods

### Overview

We compare multiple open-source LLMs on the ESPACMedlinePlus dataset, using three prompting strategies enhanced with contextual medical knowledge. Figure 1 illustrates the overall pipeline, including dataset preprocessing, prompt construction, and model evaluation.

### Data Source

- **ESPACMedlinePlus**: Parallel English-Spanish medical corpus from NIH’s MedlinePlus website, containing 2,999 aligned articles.
- Training set: 143,760 sentences after cleaning and alignment.
- Test set: 100 expert-selected sentences balanced by length.
- Additional datasets: WMT24 biomedical test set (6 language pairs) and MultiClinSum multilingual clinical summarization dataset.

### MedCOD Framework

- **LLM-KB**: Uses GPT-4o-mini to extract multilingual translations and synonyms for medical concepts.
- **UMLS**: Provides a translation dictionary and synonyms.
- Three prompt types:
  1. Multilingual translations from LLM-KB  
  2. Synonyms from LLM-KB  
  3. UMLS-based translation dictionary  
- Structured prompts incorporate this metadata to improve translation coherence and accuracy.

### Fine-tuning with LoRA

- Lightweight fine-tuning method injecting trainable weights while freezing base model parameters.
- Enables efficient adaptation of open-source LLMs to leverage MedCOD prompts.
- Reduces computational cost and storage footprint.

### Experimental Setup

- **Models**: Phi-4 (14B), Qwen2.5 (14B & 7B), Meta-LLaMA-3.1-8B, GPT-4o, GPT-4o Mini, NLLB-200 3.3B.
- **Datasets**: ESPACMedlinePlus, WMT24 biomedical test set, MultiClinSum clinical summarization.
- **Evaluation Metrics**: SacreBLEU, chrF++, COMET for translation; ROUGE and BERTScore for summarization.

---

## Results

- MedCOD + fine-tuning consistently improves translation quality across all models and metrics.
- Phi-4 (14B) with MedCOD + FT achieves the best results, surpassing GPT-4o-mini baselines.
- MedCOD generalizes well to other languages and tasks (WMT24, MultiClinSum).
- Detailed results and confidence intervals are shown in Figure 2 and Tables 1-3.

---

## Ablation Study

- Evaluates the individual and combined effects of MedCOD prompting and fine-tuning.
- Shows significant gains from both contextual augmentation and LoRA fine-tuning.
- Highlights differences in metric sensitivities (BLEU, chrF++, COMET).
- Table 4 summarizes performance across different prompt types and fine-tuning settings.

---

## Installation

```bash
git clone https://github.com/shahidul034/NoteAid-translation-EngToSpa
cd MedCOD
pip install -r requirements.txt
```

---

## Usage

---

## Usage

### Training

- **Train Qwen model:**

  ```bash
  python NoteAid-translation-EngToSpa/code/training_qwen2.py
  ```

- **Train LLaMA model:**

  ```bash
  python NoteAid-translation-EngToSpa/code/training_script.py
  ```

### Inference

- **Run inference with LLaMA model:**

  ```bash
  python NoteAid-translation-EngToSpa/code/inference_script.py
  ```

- **Run inference with NLLB model:**

  Use the Jupyter notebook for GPT-4 Mini inference and UMLS data access:

  ```
  NoteAid-translation-EngToSpa/code/gpt-4mini com.ipynb
  ```

---



---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, improvements, or new features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

- [MedlinePlus Dataset](https://medlineplus.gov)  
- LoRA: Hu et al., 2021. LoRA: Low-Rank Adaptation of Large Language Models  
- SacreBLEU, chrF++, COMET, ROUGE, BERTScore papers (see appendix for full citations)  
- Phi-4, Qwen2.5, LLaMA, GPT-4o model papers  

---

## Contact

For questions or collaboration, please contact:  
**Md Shahidul Salim** — shahidulshakib034@gmail.com

