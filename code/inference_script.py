from utils import compute_bleu_chrf
import json
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from unsloth import FastLanguageModel
file_path = "/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(file_path, 'r', encoding='utf-8') as json_file:
    original_file = json.load(json_file)
from utils2 import keyword_relationships_umls, keyword_synonymous_umls,data_import
from utils2 import kg_gpt4o_mini, synonyms_gpt4o_diff_lang,extract_keywords,gpt4o_mini_tran_func
original_file,results_data,sentence_to_prompt=data_import()
def find_cod_prompt(english_sentence):
    return sentence_to_prompt.get(english_sentence, "Prompt not found")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""

FUNCTIONS_MAP = {
    "keyword_relationships_umls": keyword_relationships_umls,
    "keyword_synonymous_umls": keyword_synonymous_umls,
    "kg_gpt4o_mini": kg_gpt4o_mini,
    "synonyms_gpt4o_diff_lang": synonyms_gpt4o_diff_lang,
    "extract_keywords": extract_keywords,
    "gpt4o_mini_tran_func": gpt4o_mini_tran_func,
    "find_cod_prompt":find_cod_prompt
}
import sys
import argparse
parser = argparse.ArgumentParser(description="Run inference function with given text.")
parser.add_argument("--function_name", type=str, choices=FUNCTIONS_MAP.keys(), help="Name of the function to run")
parser.add_argument("--model_name", type=str, help="model name")
parser.add_argument("--txt", type=str, help="context data name")
parser.add_argument("--inf_proc", type=str, help="Inference procedure")
args = parser.parse_args()

# Get the function from the map
selected_function = FUNCTIONS_MAP[args.function_name]



###########################
inference_proc=args.inf_proc  #"prompt"
def return_prompt_data(text):
    return selected_function(text)

max_seq_length,dtype,load_in_4bit,model_name = 2048,None ,True, args.model_name #"unsloth/Qwen2.5-7B-Instruct"
txt= args.txt # "without_finetune_keyword_relationships_umls"
###########################

print(f'''
###############################               
#  {model_name}
#  {txt}                         
###############################
''')
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

total_score=[]
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
results_file_path = "/home/mshahidul/project1/results_new/Medline/medlineplus_gpt4_mini_COD_back_translation.json"
with open(results_file_path, 'r', encoding='utf-8') as json_file:
    results_data = json.load(json_file)
sentence_to_prompt = {item['Original_English_sentence']: item['COD_prompt'] for item in results_data}
def find_cod_prompt(english_sentence):
    return sentence_to_prompt.get(english_sentence, "Prompt not found")
def inference(ques,pr):
    prompt=f'''
    {pr}
    Using above context translate the input into English to Spanish:
    English: {ques}
    Spanish:
    '''
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            prompt, # instruction
            ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    ans=tokenizer.batch_decode(outputs)
    start_marker = '### Response:\n'
    end_marker = '<|end_of_text|>'

    # Find the start and end positions
    start_index = ans[0].find(start_marker) + len(start_marker)
    end_index = ans[0].find(end_marker)
    response = ans[0][start_index:end_index].strip()
    return response 

def back_inference(ques,pr):
    prompt=f'''
    {pr}
    Using above context translate the input into Spanish to English:
    Spanish: {ques}
    English:
    '''
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            prompt, # instruction
            ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    ans=tokenizer.batch_decode(outputs)
    start_marker = '### Response:\n'
    end_marker = '<|end_of_text|>'

    # Find the start and end positions
    start_index = ans[0].find(start_marker) + len(start_marker)
    end_index = ans[0].find(end_marker)
    response = ans[0][start_index:end_index].strip()
    return response 

#########################################################
def inference_direct(ques):
    prompt=f'''
        Translate the input into English to Spanish:
        English: {ques}
        Spanish:"
    '''
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            prompt, # instruction
            ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    ans=tokenizer.batch_decode(outputs)
    start_marker = '### Response:\n'
    end_marker = '<|end_of_text|>'

    # Find the start and end positions
    start_index = ans[0].find(start_marker) + len(start_marker)
    end_index = ans[0].find(end_marker)
    response = ans[0][start_index:end_index].strip()
    return response 

def back_inference_direct(ques):
    prompt=f'''
        Translate the input into Spanish to English:
        Spanish: {ques}
        English:"
    '''
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            prompt, # instruction
            ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    ans=tokenizer.batch_decode(outputs)
    start_marker = '### Response:\n'
    end_marker = '<|end_of_text|>'

    # Find the start and end positions
    start_index = ans[0].find(start_marker) + len(start_marker)
    end_index = ans[0].find(end_marker)
    response = ans[0][start_index:end_index].strip()
    return response 

#########################################################


for line in tqdm.tqdm(original_file):
    try:
        context=return_prompt_data(line['english'])
        if inference_proc=="prompt":
            hypothesis_text = inference(line['english'],context)
            back_translate = back_inference(hypothesis_text,context)
        elif inference_proc=="direct":
            hypothesis_text = inference_direct(line['english'])
            back_translate = back_inference_direct(hypothesis_text)
        else:
            exit()
        score=compute_bleu_chrf(line['english'].lower(), back_translate.lower())  
        total_score.append({
            "original_english": line['english'],
            "original_spanish": line['spanish'],
            "translated_spanish": hypothesis_text,
            "back_translated_english": back_translate,
            "bleu_score": score
        })
    except Exception as e:
        print(e)
        continue

tt=model_name.split("/")[1]
avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score]) / len(total_score)
res_txt=f"{tt}_{txt}: {avg_bleu_score:.4f}"
print(res_txt)
from utils import save_to_json
save_to_json("/home/mshahidul/project1/all_tran_data/prompt_info/results2.json",res_txt) 
# with open(f"/home/mshahidul/project1/all_tran_data/sample/{tt}_{txt}.json", 'w', encoding='utf-8') as json_file:
#     json.dump(total_score, json_file, ensure_ascii=False, indent=4)

# with open(f"/home/mshahidul/project1/all_tran_data/sample/{tt}_without_finetune_and_prompt.json", 'w', encoding='utf-8') as json_file:
#     json.dump(total_score, json_file, ensure_ascii=False, indent=4)