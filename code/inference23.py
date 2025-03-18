from utils import compute_bleu_chrf
import json
import tqdm
import os
from utils2 import data_import
from unsloth import FastLanguageModel
from utils2 import keyword_relationships_umls, keyword_synonymous_umls
from utils2 import kg_gpt4o_mini, synonyms_gpt4o_diff_lang, extract_keywords, gpt4o_mini_tran_func, find_cod_prompt
import gradio as gr
from unsloth.chat_templates import get_chat_template

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
original_file, results_data, sentence_to_prompt = data_import()

FUNCTIONS_MAP = {
    "keyword_relationships_umls": keyword_relationships_umls,
    "keyword_synonymous_umls": keyword_synonymous_umls,
    "kg_gpt4o_mini": kg_gpt4o_mini,
    "synonyms_gpt4o_diff_lang": synonyms_gpt4o_diff_lang,
    "extract_keywords": extract_keywords,
    "gpt4o_mini_tran_func": gpt4o_mini_tran_func,
    "UMLS_COD_back_translation": find_cod_prompt
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

# List of available models
MODEL_LIST = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-14B-Instruct",
    "unsloth/llama-3-8b-Instruct",
    "unsloth/phi-4"
]

max_seq_length, dtype, load_in_4bit = 2048, None, True

def load_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    return model, tokenizer

import re
def extract_translation(text):
    match = re.search(r"Spanish: (.*?)<\|im_end\|>", text)
    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        text = text.split("\n")
        text = text[len(text)-1]
        text = text.split("<|im_end|>")[0]
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
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
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
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=100, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def inference_direct(text, model, tokenizer):
    prompt = f'''
        Translate the following text into Spanish: 
        {text}
        Spanish:
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def back_inference_direct(text, model, tokenizer):
    prompt = f'''
        Translate the following text into English: 
        {text}
        English:
        '''
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, temperature=1.0, min_p=1.0)
    return extract_translation(tokenizer.batch_decode(outputs)[0])

def process_input(text, function_names, model_name, inf_proc):
    # Load the selected model
    model, tokenizer = load_model(model_name)
    
    # Prepare the context based on selected functions
    context = return_prompt_data(text, function_names) if function_names else "No context provided."
    
    # Perform inference based on the selected procedure
    if inf_proc == "prompt":
        hypothesis_text = inference(text, context, model, tokenizer)
        back_translate = back_inference(hypothesis_text, context, model, tokenizer)
    else:  # direct
        hypothesis_text = inference_direct(text, model, tokenizer)
        back_translate = back_inference_direct(hypothesis_text, model, tokenizer)
    
    # Compute BLEU score
    score = compute_bleu_chrf(text.lower(), back_translate.lower())
    
    # Prepare output
    output = {
        "original_english": text,
        "translated_spanish": hypothesis_text,
        "back_translated_english": back_translate,
        "bleu_score": score['bleu_score'],
        "context_used": context
    }
    
    # return (f"Original English: {output['original_english']}\n"
    #         f"Translated Spanish: {output['translated_spanish']}\n"
    #         f"Back Translated English: {output['back_translated_english']}\n"
    #         f"BLEU Score: {output['bleu_score']:.4f}\n"
    #         f"Context Used:\n{output['context_used']}")
    return (f'''
            Back Translated English: {output['back_translated_english']}

            BLEU Score: {output['bleu_score']:.4f}

''')

# Gradio Interface
with gr.Blocks(title="Translation and Back-Translation Tool") as demo:
    gr.Markdown("## Translation and Back-Translation Tool")
    gr.Markdown("Enter text, select functions, model, and inference procedure to translate and evaluate.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input English Text", placeholder="Enter text to translate...")
            function_checkboxes = gr.CheckboxGroup(
                choices=list(FUNCTIONS_MAP.keys()),
                label="Select Functions for Context"
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_LIST,
                label="Select Model",
                value=MODEL_LIST[0]
            )
            inf_proc_radio = gr.Radio(
                choices=["direct", "prompt"],
                label="Inference Procedure",
                value="direct"
            )
            submit_button = gr.Button("Process")
        
        with gr.Column():
            output_text = gr.Textbox(label="Results", lines=10)
    
    submit_button.click(
        fn=process_input,
        inputs=[text_input, function_checkboxes, model_dropdown, inf_proc_radio],
        outputs=output_text
    )

demo.launch(share=True)