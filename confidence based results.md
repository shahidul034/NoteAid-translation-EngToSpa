### ✅ Comparison: *Old vs New Results (Without Finetune + With Context)*

| Translation Direction | BLEU (Old) | BLEU (New) | Δ BLEU | chrF++ (Old) | chrF++ (New) | Δ chrF++ |
|-----------------------|------------|------------|--------|---------------|----------------|-----------|
| en → fr               | 41.95      | 42.84      | 🔼 +0.89   | 68.06         | 68.60          | 🔼 +0.54   |
| en → de               | 22.44      | 22.09      | 🔽 -0.35   | 54.91         | 54.87          | 🔽 -0.04   |
| en → ru               | 26.28      | 25.64      | 🔽 -0.64   | 57.98         | 56.43          | 🔽 -1.55   |
| en → pt               | 45.37      | 45.60      | 🔼 +0.23   | 70.82         | 70.86          | 🔼 +0.04   |
| en → es               | 42.02      | 43.07      | 🔼 +1.05   | 67.03         | 67.50          | 🔼 +0.47   |
| en → it               | 24.93      | 25.24      | 🔼 +0.31   | 54.99         | 56.12          | 🔼 +1.13   |
| de → en               | 30.44      | 38.25      | 🔼 +7.81   | 57.06         | 65.77          | 🔼 +8.71   |
| fr → en               | 34.09      | 44.29      | 🔼 +10.20  | 57.57         | 68.63          | 🔼 +11.06  |
| ru → en               | 26.20      | 28.43      | 🔼 +2.23   | 51.10         | 53.91          | 🔼 +2.81   |
| es → en               | 37.87      | 44.41      | 🔼 +6.54   | 62.39         | 70.02          | 🔼 +7.63   |
| pt → en               | 36.43      | 43.52      | 🔼 +7.09   | 60.93         | 67.61          | 🔼 +6.68   |
| it → en               | 30.90      | 32.50      | 🔼 +1.60   | 60.58         | 61.46          | 🔼 +0.88   |

---

### 🔍 Key Observations:

- **Best Improvement in BLEU**:
  - **French → English (fr → en)** improved by **+10.20 BLEU**, and **+11.06 chrF++**.
  - **German → English (de → en)** followed closely with **+7.81 BLEU**, and **+8.71 chrF++**.

- **Slight Drops in Performance**:
  - **English → Russian (en → ru)** saw the **biggest drop in chrF++: -1.55**.
  - **English → German (en → de)** dropped slightly in both BLEU (**-0.35**) and chrF++ (**-0.04**), but is nearly identical to old performance.

- **Overall**:
  - 10 out of 12 directions improved in BLEU.
  - All but two improved in chrF++.
  - Most gains are **substantial**, especially in **de ↔ en**, **fr ↔ en**, **es ↔ en**, and **pt ↔ en** directions.


### ✅ Comparison: *Old vs New Results (With Finetune + With Context)*
| Translation Direction | Metric | Old Results | New Results | Difference |
|----------------------|--------|-------------|-------------|------------|
| en → fr | BLEU | 44.21 | 44.09 | -0.12 |
| | chrF++ | 68.85 | 68.80 | -0.05 |
| en → de | BLEU | 21.39 | 21.71 | +0.32 |
| | chrF++ | 53.35 | 52.91 | -0.44 |
| en → ru | BLEU | 27.95 | 27.81 | -0.14 |
| | chrF++ | 57.79 | 57.34 | -0.45 |
| en → pt | BLEU | 43.92 | 43.52 | -0.40 |
| | chrF++ | 69.82 | 69.15 | -0.67 |
| en → es | BLEU | 44.15 | 42.35 | -1.80 |
| | chrF++ | 67.78 | 66.20 | -1.58 |
| en → it | BLEU | 25.75 | 26.23 | +0.48 |
| | chrF++ | 56.01 | 57.13 | +1.12 |
| de → en | BLEU | 39.89 | 40.61 | +0.72 |
| | chrF++ | 66.77 | 67.35 | +0.58 |
| ru → en | BLEU | 37.03 | 38.14 | +1.11 |
| | chrF++ | 65.06 | 66.28 | +1.22 |
| es → en | BLEU | 47.08 | 47.40 | +0.32 |
| | chrF++ | 72.04 | 72.62 | +0.58 |
| fr → en | BLEU | 45.04 | 45.67 | +0.63 |
| | chrF++ | 68.42 | 68.62 | +0.20 |
| pt → en | BLEU | 46.45 | 46.58 | +0.13 |
| | chrF++ | 70.59 | 70.65 | +0.06 |
| it → en | BLEU | 33.10 | 38.15 | +5.05 |
| | chrF++ | 62.98 | 70.22 | +7.24 |

Key observations:
1. Most significant improvements are in Italian → English translation (BLEU +5.05, chrF++ +7.24)
2. Slight declines in translations to Spanish, Portuguese, and Russian
3. Generally better performance in X → English directions compared to English → X
4. Most changes are relatively small (less than 1 point) except for Italian → English
5. English → Spanish shows the largest decline (BLEU -1.80, chrF++ -1.58)


---



