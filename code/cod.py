import mysql.connector
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI  # Assuming OpenAI API is set up
from utils import get_synonyms, back_translate, compute_bleu_chrf
from openai import OpenAI 
client = OpenAI(api_key="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")
import json
def translate_using_prompt(prompt,sentence):
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                    {"role": "user", "content": f"Translate the following text from English into Spanish: {sentence}"}],
            temperature=0.5
        )
    ans=(response.choices[0].message.content)
    return ans
def back_translate(spa_tran):
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                # {"role": "system", "content": f""},
                    {"role": "user", "content": f"Translate the following text from Spanish into English: {spa_tran}"}],
            temperature=0.5
        )
    return (response.choices[0].message.content)

# Database connection details
db_config = {
    'host': '172.16.34.1',
    'port': 3307,
    'user': 'umls',
    'password': 'umls',
    'database': 'umls2024'
}

# Define the NLLB-200 model
model_name = "facebook/nllb-200-3.3B"
cache_directory = "/data/data_user_alpha/public_models"

# List of target languages
languages = {
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Portuguese": "por_Latn",
    "Spanish": 'spa_Latn'
}

# Load translation model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_directory, torch_dtype=torch.float16)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
translator = pipeline("translation", model=model, tokenizer=tokenizer)
api_key=""
# OpenAI API Client
client = OpenAI(api_key=api_key)

def extract_keywords(sentence):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract medical keywords from the given sentence. return it as python list format without extra things."},
                  {"role": "user", "content": f"{sentence}"}],
        temperature=0.5
    )
    # print(response.choices[0].message.content)
    # keywords = json.loads(response.choices[0].message.content)
    return (response.choices[0].message.content) 

def search_umls(keyword):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT CUI FROM MRCONSO WHERE STR LIKE %s LIMIT 1", (f"%{keyword}%",))
        result = cursor.fetchone()
        if not result:
            return None
        cui = result["CUI"]

        cursor.execute("SELECT LAT, STR FROM MRCONSO WHERE CUI = %s AND LAT IN (%s, %s, %s)",
                       (cui, 'FRE', 'POR', 'GER'))
        rows = cursor.fetchall()
        
        translations = {row['LAT']: row['STR'] for row in rows}
        return translations
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if connection.is_connected():
            connection.close()

# def translate_non_medical(keyword):
#     translations = {}
#     for language, lang_code in languages.items():
#         output = translator(keyword, src_lang="eng_Latn", tgt_lang=lang_code, max_length=400)
#         translations[language] = output[0]['translation_text']
#     return translations

def convert_to_chained_format(dictionary, src_lang, target_lang):
    chain = []
    
    for category, words in dictionary.items():
        for word, translations in words.items():
            formatted_translations = []
            
            # Ensure the source language is first, target language second, then others
            ordered_languages = ["ENG", "SPA", "FRE", "GER", "POR"]
            
            for lang in ordered_languages:
                if lang in translations:
                    formatted_translations.append(f"{translations[lang]}")

            chain.append(" means ".join(formatted_translations))
    
    chained_text = ". ".join(chain) + "."
    return chained_text

def process_sentence(sentence):
    keywords = extract_keywords(sentence)
    import ast
    keywords = ast.literal_eval(keywords)
    medical_translations = {}
    output=[]
    # Process medical keywords
    for keyword in keywords:
        translation={}
        translation = search_umls(keyword)
        # print(keyword)
        if translation and "SPA" not in translation:  # If Spanish missing in UMLS, use NLLB for Spanish
            translation["SPA"] = translator(keyword, src_lang="eng_Latn", tgt_lang="spa_Latn", max_length=400)[0]['translation_text']
        
        if translation:
            translation['ENG'] = keyword
            medical_translations[keyword] = translation

    result_json_temp = {
        "medical": medical_translations
    }

    src_language = "English"
    target_language = "Spanish"

    if medical_translations:
        chained_output = convert_to_chained_format(result_json_temp, src_language, target_language)
        full_prompt="Chain of dictionary: "+chained_output
    else:
        return None, None, None


    return full_prompt, medical_translations, keywords

# Example usage
# sentence= sampled_medlineplus_data[50]['english']
# print(sentence)
# sentence = "A stress fracture is a hairline crack in the bone that develops because of repeated or prolonged forces against the bone."
# full_prompt,medical_translations,keywords = process_sentence(sentence)
# print(full_prompt,medical_translations,keywords)

file_path = "/home/mshahidul/project1/all_tran_data/Sampled_100_MedlinePlus_eng_spanish_pair.json"
output_data = []
import tqdm
import json
from utils import translate_using_prompt
with open(file_path, 'r', encoding='utf-8') as json_file:
    sampled_medlineplus_data = json.load(json_file)
not_trans=[]
for x in tqdm.tqdm(sampled_medlineplus_data):

    sentence_eng=x['english']
    sentence_spa=x['spanish']
    try:
        full_prompt,chain_of_dict,keywords = process_sentence(sentence_eng)
        spa_tran=translate_using_prompt(full_prompt,sentence_eng)
        reference_text = [sentence_spa]
        hypothesis_text = spa_tran
        scores_cod_prompt = compute_bleu_chrf(reference_text, hypothesis_text)
    except:
        not_trans.append(x)
        continue

    # full_prompt,chain_of_dict,keywords = process_sentence(sentence_eng)
    
    

    output_data.append({
        "Original_English_sentence": sentence_eng,
        "Original_Spanish_sentence": sentence_spa,
        "spanish_translation": spa_tran,
        "chain_of_dict": chain_of_dict,
        "keywords": keywords,
        "scores_cod_prompt(bleu and chrf)": scores_cod_prompt,
    })
# save the output data to a JSON file
output_file = "/home/mshahidul/project1/all_tran_data/translated_MedlinePlus_100)_using_COD_prompt.json"
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)