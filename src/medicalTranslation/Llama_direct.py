import sys
import os
import torch.distributed as dist
import re
import pandas as pd
from typing import Any, Dict, List, Optional, Union
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', '127.0.0.1')
os.environ['MASTER_PORT'] = '18888'
os.environ['NCLL_DEBUG']="INFO"
os.environ['TORCH_SHOW_CPP_STACKTRACES']="1"
os.environ['TORCH_CPP_LOG_LEVEL']="INFO"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TIKTOKEN_CACHE_DIR"] = ""
dist.init_process_group(backend='gloo', init_method='env://')
sys.path.append('/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine')
from llama import Llama
from typing import List
import json
inter_example_sep="\n\n###\n\n"
intra_example_sep="\n\n"
answer_prefix="Translation: "
question_prefix="Original clinical note: "
generator = Llama.build(
        ckpt_dir="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta_Llama3.2-1B",
        tokenizer_path="/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/llama3-demo/llama3/Meta_Llama3.2-1B/tokenizer.model",
        max_seq_len=800,
        max_batch_size=1,
    )
def get_translation_content(text):
    match = re.search(r'Translation:(.*?)(\n|$)', text)
    return match.group(1).strip() if match else None
with open('/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/Sampled_100_MedlinePlus_eng_spanish_pair.json', 'r') as file:
    data = json.load(file)

def build_query_from_example(sourceNote: Union[str, List[str]], translation: Optional[str]=None) -> str:
        TEMPLATE = """Original clinical note: {sourceNote}

Translation: {translation}"""
            
        query = TEMPLATE.format(sourceNote=sourceNote, translation=translation)
        return query

instruction = (
            "Provided an original English clinical note, generate a Spanish translation of this clinical note. Desired traits for the translation are: 1) Clinical usefulness - The translation is useful and can be used in clinical settings, 2) Clinical Accuracy - The translation is clinically accurate, 3) Overall Clarity - The translation is easy to understand, 4) Coverage - The translation covers all important clinical content compared with the original note, and 5) Fluent. Response should begin with - Translation:\n\n"
        )

examples_df = pd.read_json("/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/data/tasks/ML/translated_inital.jsonl", orient="records", lines=True)
prompt = []
for i, row in examples_df.iterrows():
    prompt.append(build_query_from_example(row["Original clinical note"], row["translation"]))
prompt = instruction + inter_example_sep.join(prompt) + inter_example_sep
output_file  = "/project/pi_hongyu_umass_edu/zonghai/hospital_translation/IFT_Metrics/IFT_Metrics/data/llama3.2_1b_direct.json"  
result_data = []
def make_query(source,prompt):
    query = f"{prompt}{question_prefix}{source}{intra_example_sep}"
    return query
for sentence in data:
    prompts = make_query(sentence['english'],prompt)
    print(prompts)
    results = generator.text_completion(
        [prompts],
        max_gen_len=800,
        temperature=0.6,
        top_p=0.9,
    )
    print(get_translation_content(results[0]['generation']))
    result_data.append({
        "original_eng": sentence['english'],
        "original_spa": sentence['spanish'],
        "tran_spa": get_translation_content(results[0]['generation'])
    })
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(result_data, outfile, ensure_ascii=False, indent=4)
