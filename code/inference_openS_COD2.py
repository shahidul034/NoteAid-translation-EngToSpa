from utils import back_translate
from utils import compute_bleu_chrf
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
file_path = "/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json"
with open(file_path, 'r', encoding='utf-8') as json_file:
    original_file = json.load(json_file)

info_file_path = "/home/mshahidul/project1/all_tran_data/dataset/medlineplus_info.json"
with open(info_file_path, 'r', encoding='utf-8') as json_file:
    info_data = json.load(json_file)
    # Create a dictionary to map Original_English_sentence to synonyms
sentence_to_synonyms = {item['Original_English_sentence']: item['synonyms'] for item in info_data}

    # Function to find synonyms based on English sentence
def find_synonyms(english_sentence):
    return sentence_to_synonyms.get(english_sentence, "Synonyms not found")

results_file_path = "/home/mshahidul/project1/results_new/Medline/medlineplus_gpt4_mini_COD_back_translation.json"
with open(results_file_path, 'r', encoding='utf-8') as json_file:
    results_data = json.load(json_file)

# Create a dictionary to map Original_English_sentence to COD_prompt
sentence_to_prompt = {item['Original_English_sentence']: item['COD_prompt'] for item in results_data}

# Function to find COD_prompt based on English sentence
def find_cod_prompt(english_sentence):
    return sentence_to_prompt.get(english_sentence, "Prompt not found")


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""
max_seq_length=2048
model_name = "unsloth/Qwen2.5-14B-Instruct"
model_name2=model_name.split("/")[1]
from unsloth import FastLanguageModel
model2, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )
FastLanguageModel.for_inference(model2) # Enable native 2x faster inference

total_score=[]
FastLanguageModel.for_inference(model2) # Enable native 2x faster inference



COD_plus={
    "prompt1.1":"Expand this chain of dictionary translations by adding missing languages where appropriate: ",
    "prompt1.2": "Identify any missing translations in this language chain and complete them: ",
    "prompt2.1":"For each word in this chain of dictionary translations, add appropriate synonyms in parentheses: ",
    "prompt2.2": "Enhance this dictionary chain by inserting synonyms for each term in the respective language: ",
    "prompt3.1":"Expand this dictionary chain by adding one-hop related concepts from a knowledge graph. Include medical, biological, or psychological terms where relevant: ",
    "prompt3.2": "Enhance this chain of dictionary translations by inserting related concepts (such as medical conditions, biological processes, or symptoms) at each step: "
}

from openai import OpenAI
client = OpenAI(api_key=json.load(open('/home/mshahidul/project1/api.json', 'r'))['openai_api'])
def COD_expansion(ext_prompt):
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                # {"role": "system", "content": f""},
                    {"role": "user", "content": f"{ext_prompt}"}],
            temperature=0.5
        )
    return (response.choices[0].message.content)
from utils import save_to_json
def inference_with_prompt(ques,extension_prompt):
    prompt_ext=COD_expansion(f"{extension_prompt}\n{find_cod_prompt(ques)}")
    save_to_json(f"/home/mshahidul/project1/results_new/prompt_extention_testing.json",f"{extension_prompt}\n\n{prompt_ext}")
    # {find_synonyms(ques)}\n
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            f"{prompt_ext}\nUse this context to translate the input into English to Spanish:", # instruction
            ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model2.generate(**inputs, max_new_tokens = 64, use_cache = True)
    ans=tokenizer.batch_decode(outputs)
    start_marker = '### Response:\n'
    end_marker = '<|end_of_text|>'

    # Find the start and end positions
    start_index = ans[0].find(start_marker) + len(start_marker)
    end_index = ans[0].find(end_marker)
    response = ans[0][start_index:end_index].strip()
    return response.split("\n\n")[0]



def check_with_prompt(extension_prompt):
    import tqdm
    for line in tqdm.tqdm(original_file[:1]):
        try:
            spa_tran_prompt = inference_with_prompt(line['english'],extension_prompt)
            
            reference_text = [line['spanish']]
            hypothesis_text = spa_tran_prompt
            score1=compute_bleu_chrf(reference_text, hypothesis_text)  

            # spa_tran_direct=inference_without_prompt(line['english'])
        
            # reference_text = [line['spanish']]
            # hypothesis_text = spa_tran_direct
            # score2=compute_bleu_chrf(reference_text, hypothesis_text)

            total_score.append({
                "Original_English_sentence": line['english'],
                "Original_Spanish_sentence": line['spanish'],
                "spanish_translation_prompt": spa_tran_prompt,
                # "spanish_translation_direct": spa_tran_direct,
                "scores_cod_prompt(bleu and chrf)": score1,
                # "scores_direct(bleu and chrf)": score2
            })
        except Exception as e:
            print(f"Error processing line: {line}")
            print(e)

    # output_file_path = f"/home/mshahidul/project1/results_new/medline_{model_name2}_direct_and_COD_translation.json"
    # output_file_path = f"/home/mshahidul/project1/results_new/medline_{model_name2}_syn_translation(spa_ref).json"
    # with open(output_file_path, 'w', encoding='utf-8') as json_file: ./0.
    #     json.dump(total_score, json_file, ensure_ascii=False, indent=4)

    # Initialize variables to store the sum of scores
    total_bleu_cod_prompt = 0




    # Iterate through the total_score list to sum up the scores
    for score in total_score:
        total_bleu_cod_prompt += score["scores_cod_prompt(bleu and chrf)"]['bleu_score']
        
        # total_bleu_direct += score["scores_direct(bleu and chrf)"]['bleu_score']


    # Calculate the average scores
    num_entries = len(total_score)
    avg_bleu_cod_prompt = total_bleu_cod_prompt / num_entries

    # avg_bleu_direct = total_bleu_direct / num_entries

    txt=f"{model_name2} || prompt extention: {extension_prompt}: {avg_bleu_cod_prompt}"

    # Print the average scores
    print(txt)
    from utils import save_to_json 
    save_to_json(f"/home/mshahidul/project1/results_new/{model_name2}_prompt_extention_score.json",txt)

    
for x in COD_plus:
    check_with_prompt(COD_plus[x])
    print("\n\n")