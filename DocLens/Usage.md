# Usage introduction:
Doclen-LLM-judge evaluation:
1. First run LLM_judge.ipynb to generate two data files: one is in /data,the other one is in /result;
2. bash scripts/eval_general_claim_generation.sh $SAVENAME $REFERENCE $PROMPT_FILE
default $PROMPT_FILE=claim_evaluation/prompts/general_subclaim_generation.json, $SAVENAME file is the file generated in /result folder and $REFERENCE is in the \data folder -- After running this command, we will get claim generated.
3. bash scripts/eval_general_api_claim_entailment.sh $SAVENAME $REFERENCE $PROMPT_FILE(default in "claim_evaluation/prompts/general_claim_entail.json"). compute claim recall and claim precision.