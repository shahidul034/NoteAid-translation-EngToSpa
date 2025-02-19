import pandas as pd
import torch
import argparse

# load rouge score module
from rouge_score import rouge_scorer

# load bert-score module
from bert_score import BERTScorer
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

# load bleurt-score module
from bleurt import score
checkpoint = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/Evaluation_Metrics/ClinicalBLEURT/BLEURT-20"
# checkpoint = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/Evaluation_Metrics/ClinicalBLEURT/ClinicalBLEURT"
bleurt_scorer = score.BleurtScorer(checkpoint)
print("Loaded BLEURT-20 checkpoint!!!!")

from umls import AutomaticFactEval
umls_scorer = AutomaticFactEval()

from gpt4o_mini_as_a_judge import llm_as_a_judge_prompt, get_openai_response, get_final_score

class IFTEvalMetrics:
    def __init__(self, datapath, target_name, prediction_name):
        self.data = pd.read_csv(datapath)
        self.target_name = target_name
        self.prediction_name = prediction_name
        self.target = self.data[target_name].fillna("").tolist()
        self.generated = self.data[prediction_name].fillna("").tolist()
        self.res = {}

    def _rouge(self):

        r1 = {'p': [], 'r' : [], 'f': []}
        r2 = {'p': [], 'r' : [], 'f': []}
        rl = {'p': [], 'r' : [], 'f': []}
        
        rouge_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        for i in range(len(self.data)):
            all_rouge_scores = rouge_instance.score(self.target[i], self.generated[i])
            r1['p'].append(all_rouge_scores['rouge1'][0])
            r1['r'].append(all_rouge_scores['rouge1'][1])
            r1['f'].append(all_rouge_scores['rouge1'][2])

            r2['p'].append(all_rouge_scores['rouge2'][0])
            r2['r'].append(all_rouge_scores['rouge2'][1])
            r2['f'].append(all_rouge_scores['rouge2'][2])

            rl['p'].append(all_rouge_scores['rougeL'][0])
            rl['r'].append(all_rouge_scores['rougeL'][1])
            rl['f'].append(all_rouge_scores['rougeL'][2])

        self.res['rouge-1'] = (sum(r1['p'])/len(r1['p']), sum(r1['r'])/len(r1['r']), sum(r1['f'])/len(r1['f']))
        self.res['rouge-2'] = (sum(r2['p'])/len(r2['p']), sum(r2['r'])/len(r2['r']), sum(r2['f'])/len(r2['f']))
        self.res['rouge-l'] = (sum(rl['p'])/len(rl['p']), sum(rl['r'])/len(rl['r']), sum(rl['f'])/len(rl['f']))
    
    def _bleurt(self):
        bleurt_scores = bleurt_scorer.score(references=self.target, candidates=self.generated)
        torch.cuda.empty_cache()

        self.res['bleurt'] = sum(bleurt_scores)/len(bleurt_scores)

    def _bertscore(self):
        bert_P, bert_R, bert_F = bert_scorer.score(self.target, self.generated)
        self.res['bertscore_p'] = bert_P.mean().item()
        self.res['bertscore_r'] = bert_R.mean().item()
        self.res['bertscore_f'] = bert_F.mean().item()
        torch.cuda.empty_cache()


    
    def _umls(self):
        umls_scores = umls_scorer.run_source_concept_faithfulness(ref_sums=self.target, gen_sums=self.generated)
        self.res['umls_f'] = umls_scores['UMLS_cuis_f']

    def _llm_as_a_judge(self, judge_model='gpt-4o-mini'):
        if judge_model == 'no_judge':
            return
        
        if 'dialogue' not in self.data.columns:
            print("LLM as a judge is not applicable")
            return 
        
        self.data['judge_prompt'] = self.data.apply(lambda x: llm_as_a_judge_prompt(x['dialogue'], x[self.prediction_name]), 1)

        all_responses = []
        all_scores = []
        for i in range(len(self.data)):
            
            if i%10==0:
                print(f"##### {i} samples are processed")
            judge_prompt = self.data.iloc[i]['judge_prompt']

            try:
                response = get_openai_response(prompt=judge_prompt, model=judge_model)
            except:
                print("########## Issue with OpenAI API")
                response = ""

            all_responses.append(response)
            
            score = get_final_score(response)
            all_scores.append(score)

        self.data['gpt_4o_mini_responses'] = all_responses
        self.data['gpt_4o_mini_scores'] = all_scores

        self.res['llm_as_a_judge'] = self.data['gpt_4o_mini_scores'].describe()['mean']

    def _winrate(self):
        pass


    def run(self, judge_model='gpt-4o-mini', skip=False):

        if not skip:

            self._rouge()
            print("Rouge Computed")
            print(self.res)

            self._bleurt()
            print("BLEURT Computed")
            print(self.res)

            self._bertscore()
            print("BERTScore Computed")
            print(self.res)

            self._umls()
            print("UMLS Computed")
            print(self.res)

        self._llm_as_a_judge(judge_model=judge_model)
        print("LLM-as-a-Judge Computed")
        print(self.res)
        
        return self.res

def get_final_results(results_dict, skip=False):
    if not skip:
        print("Rouge-1-R: ", results_dict['rouge-1'][1])
        print("Rouge-L-R: ", results_dict['rouge-l'][1])
        print("Rouge-L-F: ", results_dict['rouge-l'][2])
        print("BLEURT: ", results_dict['bleurt'])
        print("BERTScore-F: ", results_dict['bertscore_f'])
        print("UMLS-F: ", results_dict['umls_f'])
    print("LLM-as-a-Judge: ", results_dict.get('llm_as_a_judge', "Not computed"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFT evaluation metrics.")
    parser.add_argument("--datapath", type=str, help="Path to the data file")
    parser.add_argument("--target_name", type=str, help="Name of the target variable")
    parser.add_argument("--prediction_name", type=str, help="Name of the predicted variable")
    parser.add_argument("--judge_model", type=str, help="judge_model")

    args = parser.parse_args()

    eval_metric = IFTEvalMetrics(
        datapath=args.datapath,
        target_name=args.target_name,
        prediction_name=args.prediction_name
        )

    results = eval_metric.run(judge_model=args.judge_model)
    print(results)
    print("#"*50)
    print(get_final_results(results))
