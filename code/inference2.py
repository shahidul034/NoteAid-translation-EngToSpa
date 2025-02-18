from utils import compute_bleu_chrf
import json
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from unsloth import FastLanguageModel
file_path = "/home/mshahidul/project1/all_tran_data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(file_path, 'r', encoding='utf-8') as json_file:
    original_file = json.load(json_file)


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
model_name="unsloth/Qwen2.5-1.5B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# exract for qwen2.5-3b/7b-instruct
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
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
def inference(text):
    messages = [
        {"role": "user", "content": f"Translate the input into English to Spanish: \n\nEnglish: {text}\n\nSpanish:"},]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,temperature = 1.0, min_p = 1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

total_score=[]
for line in tqdm.tqdm(original_file):
    try:
        hypothesis_text = inference(line['english'])
        reference_text = line['spanish']
        score=compute_bleu_chrf(reference_text, hypothesis_text)  
        total_score.append({
            "original_english": line['english'],
            "original_spanish": line['spanish'],
            "translated_spanish": hypothesis_text,
            "bleu_score": score
        })
    except Exception as e:
        print(e)
        continue

tt=model_name.split("/")[1]
avg_bleu_score = sum([x['bleu_score']['bleu_score'] for x in total_score]) / len(total_score)

print(f"{tt} without finetune --> Average BLEU Score: {avg_bleu_score:.4f}")

with open(f"/home/mshahidul/project1/results/{tt}_without_finetune.json", 'w', encoding='utf-8') as json_file:
    json.dump(total_score, json_file, ensure_ascii=False, indent=4)