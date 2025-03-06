from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
import tqdm
import re
import json
import pandas as pd
from utils import compute_bleu_chrf
from utils import save_to_json
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
def inference(text,tokenizer,model):
        messages = [
                {"role": "user", "content": f"Translate the input into English to Spanish: \n\nEnglish: {text}\n\nSpanish:"},]
        inputs = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",).to("cuda")

        outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                                temperature = 1.0, min_p = 1.0)
        temp= tokenizer.batch_decode(outputs)[0]
        return extract_translation(temp)

def inference_without_finetune(model_name):
    model_name2=model_name.split("/")[1]
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = False,
        )
    FastLanguageModel.for_inference(model)
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    file_path = "/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json"
    total_score_without_finetune=[]
    with open(file_path, 'r', encoding='utf-8') as json_file:
        original_file = json.load(json_file)
    for line in tqdm.tqdm(original_file):
            try:
                hypothesis_text = inference(line['english'],tokenizer,model)
                reference_text = line['spanish']
                score=compute_bleu_chrf(reference_text, hypothesis_text)  
                total_score_without_finetune.append({
                    "original_english": line['english'],
                    "original_spanish": line['spanish'],
                    "translated_spanish": hypothesis_text,
                    "bleu_score": score
                })
            except Exception as e:
                print(e)
                continue
    avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score_without_finetune]) / len(total_score_without_finetune)
    txt=f"{model_name2} without finetune(Medline) --> Average BLEU Score: {avg_bleu_score:.4f}"
    print(txt)
    path_temp = f'/home/mshahidul/project1/results_new/{model_name2}.json'


    save_to_json(path_temp,txt)
    with open(f"/home/mshahidul/project1/results_new/Medline/{model_name2}_without_finetuned_medline.json", 'w', encoding='utf-8') as json_file:
        json.dump(total_score_without_finetune, json_file, ensure_ascii=False, indent=4)

    total_score_without_finetune=[]
    file_path = '/home/mshahidul/project1/all_tran_data/dataset/EHR_data.xlsx'
    
    df = pd.read_excel(file_path)
    for eng, sp in tqdm.tqdm(zip(df['english'], df['spain'])):
        try:
            hypothesis_text = inference(eng, tokenizer, model)
            reference_text = sp
            score = compute_bleu_chrf(reference_text, hypothesis_text)
            total_score_without_finetune.append({
                    "original_english": eng,
                    "original_spanish": sp,
                    "translated_spanish": hypothesis_text,
                    "bleu_score": score
                })
        except Exception as e:
            print(e)
            continue

    avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score_without_finetune]) / len(total_score_without_finetune)
    txt=f"{model_name2} without finetune (EHR data) --> Average BLEU Score: {avg_bleu_score:.4f}"
    print(txt)
    
    save_to_json(path_temp,txt)
    with open(f"/home/mshahidul/project1/results_new/EHR/{model_name2}_without_finetuned_EHR_data.json", 'w', encoding='utf-8') as json_file:
        json.dump(total_score_without_finetune, json_file, ensure_ascii=False, indent=4)            
# print(extract_translation(tokenizer.batch_decode(outputs)[0]))
# inference_without_finetune("unsloth/Qwen2.5-0.5B-Instruct")