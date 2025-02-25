import re
import sys
import math
import os
import tqdm
from typing import Any, Dict, List
import pandas as pd
import json
from tqdm import tqdm
from pandarallel import pandarallel
import multiprocessing
import traceback
import argparse

pandarallel.initialize(progress_bar=True, nb_workers=25)


from src.medicalTranslation.task_init import ResponseGenTaskInit
from src.medicalTranslation.task_iterate import ResponseGenTaskIterate
from src.medicalTranslation.feedback import ResponseGenFeedback
from src.utils import retry_parse_fail_prone_cmd

import openai
import random
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

# check if orgainization is set

if os.getenv("OPENAI_ORG") is not None:
    openai.organization = os.getenv("OPENAI_ORG")

# CODEX = "gpt-4o-mini"
GPT3 = "gpt-4o-mini"
# ENGINE = CODEX#GPT3
ENGINE = GPT3

@retry_parse_fail_prone_cmd
def iterative_response(source: str, max_attempts: int) -> str:
    
    # initialize all the required components
    
    # generation of the first response
    task_init = ResponseGenTaskInit(engine=ENGINE, prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/translated_inital.jsonl")
    
    # getting feedback
    task_feedback = ResponseGenFeedback(engine=ENGINE, prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/feedback.jsonl")

    # iteratively improving the response
    task_iterate = ResponseGenTaskIterate(engine=ENGINE, prompt_examples="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/feedback.jsonl")
    
    
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
            metaoutput, translation = task_iterate(responses_to_scores=responses_to_scores, reduce_window=reduce_window)
            # exit(0)
            #context = new_context

        print(f"\n{n_attempts} Original Note> {source} \n\n Translation> {translation} - NTOKENS> {metaoutput['usage']['total_tokens']}")
        
        if metaoutput['usage']['total_tokens'] >3000:
            reduce_window +=1
            if metaoutput['usage']['total_tokens'] >3500:
                reduce_window +=1

        feedbackmetaoutput, scores = task_feedback(context=source, response=translation)
        print(f"\n{n_attempts} SCORES> {scores} - NTOKENS> {feedbackmetaoutput['usage']['total_tokens']}")
        # print(f"scores:{scores}")
        
        # scores = scores.split("Total score:")[0] + "Total score:" + scores.split("Total score:")[1].split('*')[0]
        print(f"scores:{scores}")
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
        # rtokens, ftokens = metaoutput['usage']['total_tokens'], feedbackmetaoutput['usage']['total_tokens']
        if total_score >= 0:  # only iterate over things that are improving
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

    for i, example in enumerate(data[:]):
        if max_size!=0 and count>max_size: break
        print(f"\n\n\n****Instance: {i}****\n\n")
        if 'translation' not in example: continue
        try:
            source = example["Original clinical note"]
            # print(source)
            if type(example["Original clinical note"]) is str:
                source = example["Original clinical note"].split("\n")
            if type(example["Original clinical note"]) is list:
                source = "\n".join(source[-8:])
            # print(source)
            all_responses_to_scores = iterative_response(source, max_attempts=max_attempts)
            if all_responses_to_scores is None:
                return {"result": ["FAILED"]}
            
            res = []
            scored_responses = {}
            for response, scores in all_responses_to_scores.items():
                res.append(f"{response} [score: {scores['total_score']}] \n {scores['scores']}")
                scored_responses[scores['n_attempts']]={'translation':response, 'total_score':scores['total_score']}
            # append res to example
            example['generated_translations'] = "\n------\n".join(res)
            example['scored_translations'] = scored_responses
            # print("Writing to output file:", example)
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
        default=0,
        help="Test data size (0 means all data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default='./trans.json',
        # required=True,
        help="Output file",
    )

    args = parser.parse_args()

    run_dataset(args.max_attempts, outfile=args.output, max_size=args.size)