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

import mysql.connector

# Database configuration
db_config = {
    'host': '172.16.34.1',
    'port': 3307,
    'user': 'umls',
    'password': 'umls',
    'database': 'umls2024'
}

# def get_synonyms(keyword):
#     try:
#         # Connect to the database
#         conn = mysql.connector.connect(**db_config)
#         cursor = conn.cursor()

#         # Step 1: Find CUI for the given keyword
#         cursor.execute("SELECT CUI FROM MRCONSO WHERE STR LIKE %s LIMIT 1", (f"%{keyword}%",))
#         cuis = cursor.fetchall()

#         if not cuis:
#             print("No CUI found for the keyword.")
#             return []

#         synonyms = set()

#         # Step 2: Retrieve synonyms for each CUI
#         for cui in cuis:
#             query_synonyms = """
#             SELECT DISTINCT str FROM MRCONSO 
#             WHERE cui = %s AND lat = 'ENG'
#             """
#             cursor.execute(query_synonyms, (cui[0],))
#             synonyms.update([row[0] for row in cursor.fetchall()])

#         cursor.close()
#         conn.close()

#         return list(synonyms)

#     except mysql.connector.Error as err:
#         print(f"Error: {err}")
#         return []

def get_synonyms(keyword):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        def find_synonyms(term):
            cursor.execute("SELECT CUI FROM MRCONSO WHERE STR LIKE %s LIMIT 1", (f"%{term}%",))
            cuis = cursor.fetchall()

            if not cuis:
                return []

            synonyms = set()
            for cui in cuis:
                cursor.execute("""
                    SELECT DISTINCT STR FROM MRCONSO 
                    WHERE CUI = %s AND LAT = 'ENG'
                """, (cui[0],))
                synonyms.update([row[0] for row in cursor.fetchall()])

            return list(synonyms)
        
        synonyms_results = {}
        words = keyword.split()
        for word in words:
            synonyms = find_synonyms(word)
            if synonyms:
                synonyms_results[word] = synonyms
        
        return synonyms_results if synonyms_results else {}
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return {}
    finally:
        if connection.is_connected():
            connection.close()


def extract_keywords(sentence):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract medical and non-medical keywords from the given sentence. return it as json format without extra things."},
                  {"role": "user", "content": f"{sentence}"}],
        temperature=0.5
    )
    # print(response.choices[0].message.content)
    keywords = json.loads(response.choices[0].message.content)
    return keywords  # Expected format: {"medical": ["keyword1", "keyword2"], "non_medical": ["keyword3", "keyword4"]}


def direct_translate(sentence):
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                    {"role": "user", "content": f"Translate the following text from English into Spanish: {sentence}"}],
            temperature=0.5
        )
    return (response.choices[0].message.content)

import pandas as pd

def get_keywords(sentence):
    df = pd.read_excel("/home/mshahidul/project1/testing_dataset_modified.xlsx")
    result = df.loc[df['sentence'] == sentence, 'keywords']
    return result.iloc[0] if not result.empty else None
