from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils2 import compute_bleu_chrf
from utils2 import data_import, keyword_relationships_umls, keyword_synonymous_umls, kg_gpt4o_mini, synonyms_gpt4o_diff_lang, gpt4o_mini_tran_func, UMLS_COD_back_translation

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import os
import re
import uvicorn

app = FastAPI()

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
original_file, results_data, sentence_to_prompt = data_import()

# Function mappings
FUNCTIONS_MAP = {
    "keyword_relationships_umls": keyword_relationships_umls,
    "keyword_synonymous_umls": keyword_synonymous_umls,
    "kg_gpt4o_mini": kg_gpt4o_mini,
    "synonyms_gpt4o_diff_lang": synonyms_gpt4o_diff_lang,
    # "extract_keywords": extract_keywords,
    "gpt4o_mini_tran_func": gpt4o_mini_tran_func,
    "UMLS_COD_back_translation": UMLS_COD_back_translation
}

FUNCTIONS_MAP_info = {
    "keyword_relationships_umls": "Below is an overview of the relationships between key medical terms and their associated concepts, detailing how each term connects to medical ideas: ",
    "keyword_synonymous_umls": "Below is an overview of synonyms for each keyword in different languages: ",
    "kg_gpt4o_mini": "Below is a knowledge graph of the medical keywords in the sentence: ",
    "synonyms_gpt4o_diff_lang": "Below is an overview of synonyms for each keyword in different languages: ",
    "extract_keywords": "Below are the keywords extracted from the sentence: ",
    "gpt4o_mini_tran_func": "Below is an overview of translations for each keyword in different languages: ",
    "UMLS_COD_back_translation": "Below is a chain of dictionary entries for keywords extracted from a sentence, presented in multiple languages: "
}

MODEL_LIST = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-14B-Instruct",
    "unsloth/llama-3-8b-Instruct",
    "unsloth/phi-4"
]

max_seq_length, dtype, load_in_4bit = 2048, None, True

# Global variables to store the loaded model and tokenizer
model = None
tokenizer = None
current_model_name = None

class TranslationRequest(BaseModel):
    text: str
    function_names: List[str]
    model_name: str
    inf_proc: str
    context: str = None  
    instruction:str

def load_model(model_name):
    global model, tokenizer, current_model_name
    if model is None or current_model_name != model_name:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False
        )
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        FastLanguageModel.for_inference(model)
        current_model_name = model_name
    return model, tokenizer

def extract_translation(text):
    match = re.search(r"Spanish: (.*?)<\|im_end\|>", text)
    if match:
        return match.group(1)
    text = text.split("\n")[-1].split("<|im_end|>")[0]
    return text

def return_prompt_data(text, selected_functions):
    results = []
    for func_name in selected_functions:
        try:
            func = FUNCTIONS_MAP[func_name]
            results.append(FUNCTIONS_MAP_info[func_name] + "\n" + func(text))
        except Exception as e:
            print(f"Error: {str(e)}")
    return "\n".join(results)

def inference(text, context, model, tokenizer):
    prompt = f'''
        {context}
        Using above context translate the following text into Spanish:
        {text}
        Spanish: "
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def back_inference(text, context, model, tokenizer):
    prompt = f'''
        {context}
        Using above context translate the following text into English:
        {text}
        English: "
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=100, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def manual_inference(text, cont,inst, model, tokenizer):
    prompt = f'''
        {cont}
        {inst}
        {text}
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def manual_back_inference(text, cont,inst, model, tokenizer):
    prompt = f'''
        {cont}
        {inst}
        {text}
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=100, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def inference_direct(text, model, tokenizer):
    prompt = f'''
        Translate the following text into Spanish: 
        {text}
        Spanish:
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def back_inference_direct(text, model, tokenizer):
    prompt = f'''
        Translate the following text into English: 
        {text}
        English:
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def swap_words(s, word1, word2):
    words = s.split()
    for i in range(len(words)):
        if words[i] == word1:
            words[i] = word2
        elif words[i] == word2:
            words[i] = word1
    return ' '.join(words)

# Startup event to load the initial model
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    # Load a default model at startup (e.g., the first in the list)
    default_model = MODEL_LIST[0]
    model, tokenizer = load_model(default_model)
    print(f"Loaded default model: {default_model}")

@app.post("/process")
async def process_input(request: TranslationRequest):
    if request.model_name not in MODEL_LIST:
        raise HTTPException(status_code=400, detail="Invalid model name")
    if request.inf_proc not in ["direct", "prompt", "manual"]:
        raise HTTPException(status_code=400, detail="Invalid inference procedure")

    # Load or reuse the model and tokenizer
    global model, tokenizer
    model, tokenizer = load_model(request.model_name)

    context_temp = return_prompt_data(request.text, request.function_names) if request.function_names else "No context provided."

    if request.inf_proc == "prompt":
        hypothesis_text = inference(request.text, context_temp, model, tokenizer)
        back_translate = back_inference(hypothesis_text, context_temp, model, tokenizer)
    elif request.inf_proc == "direct":
        hypothesis_text = inference_direct(request.text, model, tokenizer)
        back_translate = back_inference_direct(hypothesis_text, model, tokenizer)
    else:
        if not request.context and not request.instruction:
            raise HTTPException(status_code=400, detail="Manual prompt is required for manual inference")
        prompt_temp_rev = swap_words(request.instruction, "English", "Spanish")
        hypothesis_text = manual_inference(request.text, request.context,request.instruction, model, tokenizer)
        back_translate = manual_back_inference(hypothesis_text, prompt_temp_rev,request.instruction, model, tokenizer)

    score = compute_bleu_chrf(request.text.lower(), back_translate.lower())

    return {
        "original_english": request.text,
        "translated_spanish": hypothesis_text,
        "back_translated_english": back_translate,
        "bleu_score": score['bleu_score'],
        "context_used": context_temp
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)