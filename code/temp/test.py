import json
updated_keywords_path = "/home/mshahidul/project1/all_tran_data/prompt_info/updated_keywords.json"
with open(updated_keywords_path, 'r', encoding='utf-8') as updated_keywords_file:
    updated_keywords = json.load(updated_keywords_file)

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

print(updated_keywords[0]['sentence'])
print(updated_keywords[0]['keywords'])
print(synonyms_gpt4o_diff_lang(updated_keywords[0]['sentence']))