import re
import sys
sys.path.append('/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine')
import math
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import os
from typing import Any, Dict, List
import pandas as pd
import json
from tqdm import tqdm
from pandarallel import pandarallel
import traceback
import argparse
import torch
from typing import List
import fire
from llama import Llama
# pandarallel.initialize(progress_bar=True, nb_workers=1)
import torch.distributed as dist
from src.medicalTranslation.task_init_LLAMA import ResponseGenTaskInit
from src.medicalTranslation.task_iterate_LLAMA import ResponseGenTaskIterate
from src.medicalTranslation.feedback_LLAMA import ResponseGenFeedback
from src.utils import retry_parse_fail_prone_cmd
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
# from transformers import LlamaForCausalLM, LlamaTokenizer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', '127.0.0.1')
os.environ['MASTER_PORT'] = '12345'
os.environ['NCLL_DEBUG']="INFO"
os.environ['TORCH_SHOW_CPP_STACKTRACES']="1"
os.environ['TORCH_CPP_LOG_LEVEL']="INFO"
os.environ["TIKTOKEN_CACHE_DIR"] = ""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dist.init_process_group(backend='gloo', init_method='env://')
# Initialize LLAMA model
# model_name = "/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/LLAMA_7b"
# tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
# model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=True, torch_dtype=torch.bfloat16).cpu()
engine = Llama.build(
            ckpt_dir="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta_Llama3.2-3B",
            tokenizer_path="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta_Llama3.2-1B/tokenizer.model",
            max_batch_size=1,
            max_seq_len=2000,
            )
# model_name = "./llama3-demo/llama3/Llama-3.2-1B"
# model_name ="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Llama-3.2-1B"
# engine = LlamaForCausalLM.from_pretrained(model_name).cuda()
# tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
@retry_parse_fail_prone_cmd
def iterative_response(source: str, max_attempts: int) -> str:
    # max_tokens = 800
    # initialize all the required components
    
    # generation of the first response
    task_init = ResponseGenTaskInit(engine = engine,prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/translated_inital.jsonl")
    
    # getting feedback
    task_feedback = ResponseGenFeedback(engine = engine,prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/feedback.jsonl")

    # iteratively improving the response
    task_iterate = ResponseGenTaskIterate(engine = engine,prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/feedback.jsonl")
    
    
    # Initialize the task

    n_attempts = 0
    
    responses_to_scores = dict()
    
    all_responses_to_scores = dict()
    best_score_so_far = 0
    reduce_window = 0
    while n_attempts < max_attempts:

        if n_attempts == 0:
            metaoutput, translation = task_init(source=source)
        else:
            # print(f"responses_to_scores: {responses_to_scores}")
            metaoutput, translation = task_iterate(responses_to_scores=responses_to_scores, reduce_window=reduce_window)

        output_string = f"\n{n_attempts} Original Note> {source} \n\n Translation> {translation} - NTOKENS> {metaoutput}"
        with open("./output_llama_5.txt", "a") as file:
                file.write(output_string)
        # print(f"\n{n_attempts} Original Note> {source} \n\n Translation> {translation}")
        if metaoutput >3000:
            reduce_window +=1
            if metaoutput >3500:
                reduce_window +=1

        feedbackmetaoutput, scores = task_feedback(context=source, response=translation)
        output_string = f"\n{n_attempts} SCORES> {scores} - NTOKENS> {feedbackmetaoutput}"
        with open("output_llama_5.txt", "a") as file:
                file.write(str(feedbackmetaoutput)+'\n')
        score_match = re.search(r"Total score: (\d+)/(\d+)", scores)
        if not score_match:
            continue
        total_score = re.search(r"Total score: (\d+)/(\d+)", scores).group(0)
        
        total_score = int(total_score.split(":")[1].strip().split("/")[0])
        
        all_responses_to_scores[translation] = {
            "n_attempts": n_attempts,
            "scores": scores,
            "total_score": total_score,
            "source": source,
        }
        
        if total_score > best_score_so_far:  # only iterate over things that are improving
            best_score_so_far = total_score
            
            responses_to_scores[translation] = (source, scores)
            
        else:
            print(f"Score of {translation} is {total_score}, which is less than the current best of {best_score_so_far}")

        n_attempts += 1
    return all_responses_to_scores



def run_dataset(max_attempts: int, outfile: str, max_size: int = 1):

    f = open('/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/fed_data.json')
    data = json.load(f)
    print('len of data', len(data))
    count=0
    outwriter = open(outfile, 'a')
    total_token = 0
    for i, example in enumerate(data[:]):
        if max_size!=0 and count>max_size: break
        print(f"\n\n\n****Instance: {i}****\n\n")
        if 'translation' not in example: continue
        try:
            source = example["Original clinical note"]
            all_responses_to_scores = iterative_response(source, max_attempts=max_attempts)
            if all_responses_to_scores is None:
                return {"result": ["FAILED"]}
            
            res = []
            scored_responses = {}
            for response, scores in all_responses_to_scores.items():
                res.append(f"{response} [score: {scores['total_score']}] \n {scores['scores']}")
                scored_responses[scores['n_attempts']]={'translation':response, 'total_score':scores['total_score']}
            example['generated_translations'] = "\n------\n".join(res)
            example['scored_translations'] = scored_responses
            outwriter.write(json.dumps(example)+'\n')
            print("\n ------ \n ".join(res))
        except Exception as e:
            print(f"error in {example}\n\n{e}", file=sys.stderr)
            traceback.print_exc()
            return {"result": ["FAILED"]}
        count+=1
    
    outwriter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Max attempts",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=3,
        help="Test data size (0 means all data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default='./trans_temp.json',
        help="Output file",
    )

    args = parser.parse_args()

    run_dataset(args.max_attempts, outfile=args.output, max_size=args.size)