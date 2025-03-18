import json
keyword_relationships_path = "/home/mshahidul/project1/all_tran_data/prompt_info/keyword_relationships_umls.json"
with open(keyword_relationships_path, 'r', encoding='utf-8') as keyword_relationships_file:
    keyword_relationships = json.load(keyword_relationships_file)
# 1
def keyword_relationships_umls(sentence):
    for keyword_relationship in keyword_relationships:
        if keyword_relationship['sentence'] == sentence:
            res=keyword_relationship['relationships']
            list=[]
            for k,v in res.items():
                if len(v):
                    list.append(f"{k} is a {v[0]['relation']} to the term '{v[0]['related_concept']}'")
            full_list = ', '.join(list)
            return full_list
    return None

# 2
import json
keyword_synonymous_path = "/home/mshahidul/project1/all_tran_data/prompt_info/keyword_synonymous_umls.json"
with open(keyword_synonymous_path, 'r', encoding='utf-8') as keyword_synonymous_file:
    keyword_synonymous = json.load(keyword_synonymous_file)
def keyword_synonymous_umls(sentence):
    for keyword_synonym in keyword_synonymous:
        if keyword_synonym['sentence'] == sentence:
            res=keyword_synonym['synonymous']
            list=[]
            for k,v in res.items():
                if len(v):
                    v['POR']= v['POR'] if v.get('POR') is not None else "None"
                    v['FRE']= v['FRE'] if v.get('FRE') is not None else "None"
                    v['GER']= v['GER'] if v.get('GER') is not None else "None"
                    list.append(f"The term {k} is synonymous with:\nPortuguese: '{v['POR']}'; French: '{v['FRE']}'; German: '{v['GER']}'")
            full_list = '\n'.join(list)
            return full_list
    return None

# 3
import json
kg_gpt4o_mini_path = "/home/mshahidul/project1/all_tran_data/prompt_info/kg_gpt4o_mini.json"
with open(kg_gpt4o_mini_path, 'r', encoding='utf-8') as kg_gpt4o_mini_file:
    kg_gpt4o_mini_data = json.load(kg_gpt4o_mini_file)
with open("/home/mshahidul/project1/all_tran_data/prompt_info/updated_keywords.json", 'r', encoding='utf-8') as updated_keywords_file:
    updated_keywords = json.load(updated_keywords_file)
def kg_gpt4o_mini(sentence):
    
    # keywords=None
    for x2 in updated_keywords:
        if x2['sentence']==sentence:
            keywords=x2['keywords']
            break
    full_list=[]
    for kg in kg_gpt4o_mini_data:
        if kg['keyword'] in keywords:
            list=[]
            data=kg['kg']
            if data==None:
                continue
            # print(data)
            for k,v in data.items():
                if k=="keyword":
                    continue
                if isinstance(v, str):
                    value=v
                elif isinstance(v,dict):
                    for d in v:
                        value=v[d]
                        break
                else:
                
                    value = v[0]
                    
                list.append(f"'{data['keyword']}' term {k} is '{value}'")
            full_list.append(';'.join(list))
    return "\n".join(full_list)

# 4
synonyms_gpt4o_diff_lang_path = "/home/mshahidul/project1/all_tran_data/prompt_info/synonyms_gpt4o_diff_lang.json"
with open(synonyms_gpt4o_diff_lang_path, 'r', encoding='utf-8') as synonyms_gpt4o_diff_lang_file:
    synonyms_gpt4o_diff_lang_data = json.load(synonyms_gpt4o_diff_lang_file)
sentence_to_keywords={}
for x2 in updated_keywords:
    sentence_to_keywords[x2['sentence']]=x2['keywords']


def synonyms_gpt4o_diff_lang(sentence):
    
    keywords=sentence_to_keywords[sentence]
    
    full_list=[]
    for word in synonyms_gpt4o_diff_lang_data:
        if word['keyword'] in keywords:
            data=word['synonyms']
            list=[]
            for k,v in data.items():
                 list.append(f"{k}: [{', '.join(v)}]")
            full_list.append(f"Synonyms of '{word['keyword']} in different languages':\n"+'; '.join(list))
    return "\n".join(full_list)


# 5
translated_keywords_path = "/home/mshahidul/project1/all_tran_data/prompt_info/translated_keywords_gpt4o_mini.json"
import json
with open(translated_keywords_path, 'r', encoding='utf-8') as translated_keywords_file:
    translated_keywords = json.load(translated_keywords_file)
gpt4o_mini_tran={}
for x in translated_keywords:
    gpt4o_mini_tran[x['sentence']]=x['translated_keywords']

def gpt4o_mini_tran_func(sentence):
    res=gpt4o_mini_tran[sentence]
    full_list=[]
    for k, v in res.items():
        dat=[]
        for k2,v2 in v.items():
            # if k2!='Spanish':
            dat.append(f'{k2}: "{v2}"')
        full_list.append(f"- {k}: {', '.join(dat)}")
    if len(full_list)==0:
        return "none"
    return "\n".join(full_list)   



#6
results_file_path = "/home/mshahidul/project1/results_new/Medline/medlineplus_gpt4_mini_COD_back_translation.json"
with open(results_file_path, 'r', encoding='utf-8') as json_file:
        results_data = json.load(json_file)
sentence_to_prompt = {item['Original_English_sentence']: item['COD_prompt'] for item in results_data}
def UMLS_COD_back_translation(english_sentence):
    return sentence_to_prompt.get(english_sentence, "Prompt not found")


# Extra

import sacrebleu
import evaluate
metric = evaluate.load("sacrebleu")
def compute_bleu_chrf(reference, hypothesis):
    """
    Computes the BLEU and chrF++ scores for a given reference and hypothesis.
    
    :param reference: List of reference translations (list of strings)
    :param hypothesis: The hypothesis translation (a single string)
    :return: A dictionary containing BLEU and chrF++ scores
    """
    # Ensure reference is wrapped in a list as sacrebleu expects a list of references
    # bleu_score = sacrebleu.corpus_bleu(hypothesis, [reference]).score
    # bleu_score = sacrebleu.corpus_bleu(hypothesis, [reference], tokenize="13a", lowercase=True).score
    bleu_score=metric.compute(predictions=[hypothesis], references=[reference])
    chrf_score = sacrebleu.corpus_chrf(hypothesis, [reference]).score

    return {"bleu_score": bleu_score['score'],"chrF++": chrf_score}

def data_import():
    file_path = "/home/mshahidul/project1/all_tran_data/dataset/Sampled_100_MedlinePlus_eng_spanish_pair.json"
    with open(file_path, 'r', encoding='utf-8') as json_file:
        original_file = json.load(json_file)
    results_file_path = "/home/mshahidul/project1/results_new/Medline/medlineplus_gpt4_mini_COD_back_translation.json"
    with open(results_file_path, 'r', encoding='utf-8') as json_file:
        results_data = json.load(json_file)
    sentence_to_prompt = {item['Original_English_sentence']: item['COD_prompt'] for item in results_data}
    return original_file,results_data,sentence_to_prompt

## Extra
updated_keywords_path = "/home/mshahidul/project1/all_tran_data/prompt_info/updated_keywords.json"
with open(updated_keywords_path, 'r', encoding='utf-8') as updated_keywords_file:
    updated_keywords = json.load(updated_keywords_file)

# def extract_keywords(sentence):
    # for updated_keyword in updated_keywords:
    #     if updated_keyword['sentence'] == sentence:
    #         res=updated_keyword['keywords']
    #         return res
    # return None
