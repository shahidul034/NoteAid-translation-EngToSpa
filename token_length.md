
### 🔍 **Sentence Length Statistics Across Language Pairs**

**Tokenizer:** `unsloth/Qwen2.5-14B-Instruct`
**Dataset Size:** 50 examples per language pair

| Language Pair | Avg. Input Length (chars) | Avg. Output Length (tokens) | 95th %ile Output Tokens | Max Output Tokens |
| ------------- | ------------------------- | --------------------------- | ----------------------- | ----------------- |
| en → fr       | 1317.56                   | 456.52                      | 746                     | 930               |
| fr → en       | 1726.62                   | 310.68                      | 471                     | 1634              |
| en → de       | 1428.88                   | 470.90                      | 787                     | 938               |
| de → en       | 1465.38                   | 278.08                      | 441                     | 588               |
| en → it       | 1504.04                   | 511.24                      | 824                     | 1072              |
| it → en       | 1595.20                   | 290.30                      | 566                     | 702               |
| en → es       | 1340.70                   | 429.12                      | 652                     | 749               |
| es → en       | 1909.44                   | 356.42                      | 546                     | 2124              |
| en → ru       | 1305.20                   | 459.46                      | 779                     | 871               |
| ru → en       | 1225.98                   | 243.68                      | 465                     | 555               |
| en → pt       | 1341.94                   | 419.32                      | 654                     | 727               |
| pt → en       | 1382.54                   | 294.42                      | 450                     | 499               |

---

### 📌 Notes:

* **Avg. Input Length (chars):** Average character length of source sentences.
* **Avg. Output Length (tokens):** Average number of tokens produced for target sentences using the `unsloth/Qwen2.5-14B-Instruct` tokenizer.
* **95th Percentile:** Helps configure `max_new_tokens` during inference to prevent truncation.
* **Max Tokens:** Maximum observed target token length for this sample size.

