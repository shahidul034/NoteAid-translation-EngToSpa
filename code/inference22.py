from utils2 import compute_bleu_chrf
import json
import tqdm
import os
from utils2 import data_import
from unsloth import FastLanguageModel
from utils2 import keyword_relationships_umls, keyword_synonymous_umls
from utils2 import kg_gpt4o_mini, synonyms_gpt4o_diff_lang,extract_keywords,gpt4o_mini_tran_func,UMLS_COD_back_translation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
original_file,results_data,sentence_to_prompt=data_import()

FUNCTIONS_MAP = {
    "keyword_relationships_umls": keyword_relationships_umls,
    "keyword_synonymous_umls": keyword_synonymous_umls,
    "kg_gpt4o_mini": kg_gpt4o_mini,
    "synonyms_gpt4o_diff_lang": synonyms_gpt4o_diff_lang,
    "extract_keywords": extract_keywords,
    "gpt4o_mini_tran_func": gpt4o_mini_tran_func,
    "UMLS_COD_back_translation":UMLS_COD_back_translation
}
FUNCTIONS_MAP_info = {
    "keyword_relationships_umls": "Below is an overview of the relationships between key medical terms and their associated concepts, detailing how each term connects to medical ideas: ",
    "keyword_synonymous_umls": "Below is an overview of synonyms for each keyword in different languages: ",
    "kg_gpt4o_mini": "Below is a knowledge graph of the medical keywords in the sentence: ",
    "synonyms_gpt4o_diff_lang": "Below is an overview of synonyms for each keyword in different languages: ",
    "extract_keywords": "Below are the keywords extracted from the sentence: ",
    "gpt4o_mini_tran_func": "Below is an overview of translations for each keyword in different languages: ",
    "UMLS_COD_back_translation":"Below is a chain of dictionary entries for keywords extracted from a sentence, presented in multiple languages: "
}

import sys
import argparse
parser = argparse.ArgumentParser(description="Run inference function with given text.")
parser.add_argument("--function_names", type=str, nargs='+', choices=FUNCTIONS_MAP.keys(), help="Names of the functions to run (space-separated)")
parser.add_argument("--model_name", type=str, help="model name")
# parser.add_argument("--txt", type=str, help="context data name")
parser.add_argument("--inf_proc", type=str, help="Inference procedure")
args = parser.parse_args()

# Get the function from the map
selected_functions = [FUNCTIONS_MAP[func] for func in args.function_names]

txt= "without_finetune_"+("_".join([func.__name__ for func in selected_functions]))



###########################
inference_proc=args.inf_proc  #"prompt"

def return_prompt_data(text):
    results = []
    
    for func in selected_functions:
            if (func(text))==None:
                func_txt="No information"
            else:
                func_txt=func(text)
            results.append(FUNCTIONS_MAP_info[func.__name__]+"\n"+func_txt)
    
    return "\n".join(results)

max_seq_length,dtype,load_in_4bit,model_name = 2048,None ,True, args.model_name #"unsloth/Qwen2.5-7B-Instruct"
# txt= args.txt # "without_finetune_keyword_relationships_umls"
###########################


print(f'''
###############################               
#  {model_name}
#  {txt}                         
###############################
''')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)
FastLanguageModel.for_inference(model) 
import re
def extract_translation(text):
    match = re.search(r"Spanish: (.*?)<\|im_end\|>", text)
    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        text=text.split("\n")
        text=text[len(text)-1]
        text=text.split("<|im_end|>")[0]
        return text
    
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

FastLanguageModel.for_inference(model) 

def inference(text,context):
    # print (f"{keyword_relationships_umls(text)}\nUsing above context translate the input into English to Spanish: \n\nEnglish: {text}\n\nSpanish: ")
    # exit()
    # prompt=f'''
    #     {context}
    #     Using above context translate the following text into Spanish:
    #     {text}
    #     Spanish: "
    #     '''
    prompt=f'''
        {context}
        Using the above context to translate the following English text into Spanish naturally and accurately: 
        {text}"
        '''
    # prompt = f"""
    #         {context}
    #         You are a highly accurate machine translation system based on Qwen2.5. Please translate the following English text into Spanish, ensuring that all nuances, idiomatic expressions, and technical terms are accurately preserved.
    #         English: {text}
    #         Spanish: 
    #         """
    # prompt=f''' 
    #     Translate the following text into Spanish, given this context:
    #     {keyword_relationships_umls(text)}
    #     Text: {text}
    #     Spanish: "
    #     '''
    messages = [
        {"role": "user", "content": prompt,}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,temperature = 1.0, min_p = 1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def back_inference(text,context):
    # prompt=f'''
    #     {context}
    #     Using above context translate the following text into English:
    #     {text}
    #     English: "
    #     '''
    # prompt = f"""
    #         {context}
    #         You are a highly accurate machine translation system based on Qwen2.5. Please translate the following Spanish text back into English, ensuring that the original meaning, nuances, and technical details are fully maintained.
    #         Spanish: {text}
    #         English: 
    #         """
    prompt=f'''
        {context}
        Using the above context to translate the following Spanish text into English naturally and accurately: 
        {text}"
        '''
    # prompt=f'''
    #     Translate the following text into English, given this context:
    #     {keyword_relationships_umls(text)}
    #     Text: {text}
    #     English: "
    #     '''
    messages = [
        {"role": "user", "content": prompt},]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 100, use_cache = True,temperature = 1.0, min_p = 1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

##################################################


def inference_direct(text):
    prompt=f'''
            Translate the following text into Spanish: 
            {text}
            Spanish:
            '''
    messages = [
        {"role": "user", "content": prompt},]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,temperature = 1.0, min_p = 1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def back_inference_direct(text):
    prompt=f'''
            Translate the following text into English: 
            {text}
            English:
            '''
    messages = [
        {"role": "user", "content": prompt},]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,temperature = 1.0, min_p = 1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

##################################################

total_score=[]
flag=1
for line in tqdm.tqdm(original_file):
    try:
        context=return_prompt_data(line['english'])
        if flag:
            flag=0
            print("*"*30)
            print(context)
            print("*"*30)
        # context=keyword_synonymous_umls(line["english"])
        if inference_proc=="prompt":
            hypothesis_text = inference(line['english'],context)
            back_translate = back_inference(hypothesis_text,context)
        elif inference_proc=="direct":
            hypothesis_text = inference_direct(line['english'])
            back_translate = back_inference_direct(hypothesis_text)
        else:
            exit()
        # print(f"yes: {back_translate}")
        score=compute_bleu_chrf(line['english'].lower(), back_translate.lower())  
        total_score.append({
            "original_english": line['english'],
            "original_spanish": line['spanish'],
            "translated_spanish": hypothesis_text,
            "back_translated_english": back_translate,
            "bleu_score": score
        })
    except Exception as e:
        print("Error: ",e)
        continue

tt=model_name.split("/")[1]
avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score]) / len(total_score)
res_txt=f"{tt}_{txt}: {avg_bleu_score:.4f}"
print(res_txt)

with open(f"/home/mshahidul/project1/all_tran_data/sample/{tt}_{txt}.json", 'w', encoding='utf-8') as json_file:
    json.dump(total_score, json_file, ensure_ascii=False, indent=4)

from utils import save_to_json
save_to_json("/home/mshahidul/project1/all_tran_data/prompt_info/results3.json",res_txt) 
# with open(f"/home/mshahidul/project1/all_tran_data/sample/{tt}_without_finetune_and_prompt.json", 'w', encoding='utf-8') as json_file:
#     json.dump(total_score, json_file, ensure_ascii=False, indent=4)