import gradio as gr
import requests
import sys
import argparse
parser = argparse.ArgumentParser(description="Run inference function with given text.")
parser.add_argument("--p", type=str, help="port")
args = parser.parse_args()
# FastAPI backend URL
BACKEND_URL = f"http://localhost:{args.p}/process"

# List of available functions (must match FUNCTIONS_MAP in backend)
FUNCTION_LIST = [
    "keyword_relationships_umls",
    "keyword_synonymous_umls",
    "kg_gpt4o_mini",
    "synonyms_gpt4o_diff_lang",
    "extract_keywords",
    "gpt4o_mini_tran_func",
    "UMLS_COD_back_translation"
]

# List of available models (must match MODEL_LIST in backend)
MODEL_LIST = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-14B-Instruct",
    "unsloth/llama-3-8b-Instruct",
    "unsloth/phi-4"
]

def process_input(text, function_names, model_name, inf_proc,context,instruction):
    try:
        # Prepare request payload
        payload = {
            "text": text,
            "function_names": function_names,
            "model_name": model_name,
            "inf_proc": inf_proc,
            "context": context,
            "instruction":instruction
        }
        
        # Send request to FastAPI backend
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return [result['original_english'],result['back_translated_english'],result['bleu_score']]
        # return f'''
        #     Original sentence:
        #     {result['original_english']}\n
        #     Back Translated English: 
        #     {result['back_translated_english']}\n

        #     BLEU Score: {result['bleu_score']:.4f}
        # '''
        
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to connect to backend - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Translation and Back-Translation Tool") as demo:
    gr.Markdown("## Translation and Back-Translation Tool")
    gr.Markdown("Enter text, select functions, model, and inference procedure to translate and evaluate.")
    
    with gr.Row():
        with gr.Column():
            context=gr.Textbox(label="Context", placeholder="Context",value="")
            instruction=gr.Textbox(label="Manual prompting", placeholder="Prompting",value="Translate the following English text into Spanish naturally and accurately: ")
            text_input = gr.Textbox(label="Input English Text", placeholder="Enter text to translate...",value="If you have a weakened immune system due to AIDS, cancer, transplantation, or corticosteroid use, call your doctor if you develop a cough, fever, or shortness of breath.")
            function_checkboxes = gr.CheckboxGroup(
                choices=FUNCTION_LIST,
                label="Select Functions for Context"
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_LIST,
                label="Select Model",
                value=MODEL_LIST[5]
            )
            inf_proc_radio = gr.Radio(
                choices=["prompt","direct","manual"],
                label="Inference Procedure",
                value="manual"
            )
            submit_button = gr.Button("Process")
        
        with gr.Column():
            output_text1 = gr.Label(label="Original Sentence")
            output_text2 = gr.Label(label="Back translated Sentence")
            output_text3 = gr.Label(label="Bleu score")

    
    submit_button.click(
        fn=process_input,
        inputs=[text_input, function_checkboxes, model_dropdown, inf_proc_radio,context,instruction],
        outputs=[output_text1,output_text2,output_text3]
    )
    from utils2 import data_import, keyword_relationships_umls, keyword_synonymous_umls, kg_gpt4o_mini, synonyms_gpt4o_diff_lang, extract_keywords, gpt4o_mini_tran_func, find_cod_prompt
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
    def return_prompt_data(text, function_names):
        results = []
        selected_functions = [FUNCTIONS_MAP[func] for func in function_names]
        for func in selected_functions:
                if (func(text))==None:
                    func_txt="No information"
                else:
                    func_txt=func(text)
                results.append(FUNCTIONS_MAP_info[func.__name__]+"\n"+func_txt)
        
        return "\n".join(results)
    
    def update_context(function_names, input_text):
        if not function_names:
            return "No context provided."
        return return_prompt_data(input_text, function_names)
    function_checkboxes.change(
        fn=update_context,
        inputs=[function_checkboxes, text_input],
        outputs=[context]
    )
    import json
    # Load examples from JSON file
    with open('/home/mshahidul/project1/all_tran_data/prompt_info/UMLS_COD_back_translation.json', 'r') as f:
        example_data = json.load(f)

    examples = gr.Examples(
        examples=[[x['sentence'],x['COD_prompt']] for x in example_data],
        inputs=[text_input,context],
    )

demo.launch(share=True)