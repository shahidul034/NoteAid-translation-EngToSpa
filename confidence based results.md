## Qwen 2.5 with finetune and context:
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