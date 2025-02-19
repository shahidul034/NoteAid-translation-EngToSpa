# README
Doclen-LLM-judge evaluation:
1. First run DocLens/LLM_judge.ipynb to generate two data files: generation.json is in \data, reference.json is in \result;
2. generate claims
```bash
bash scripts/eval_general_claim_generation.sh $SAVENAME $REFERENCE $PROMPT_FILE
```
default $PROMPT_FILE=claim_evaluation/prompts/general_subclaim_generation.json, $SAVENAME file=\DocLens\result\generation.json, $REFERENCE=\DocLens\data\reference.json  
3. Claim Recall and Claim Precision Computation
```bash
bash scripts/eval_general_api_claim_entailment.sh $SAVENAME $REFERENCE $PROMPT_FILE
```
default prompt file is in "claim_evaluation/prompts/general_claim_entail.json". 
Traditional evaluation matrices:
/IFT_Metrics/IFT_Metrics/EvalTranslation.ipynb
