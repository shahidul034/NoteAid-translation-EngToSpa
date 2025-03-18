from openai import OpenAI
api_key="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" 
client = OpenAI(api_key=api_key)
import json

def translate_using_prompt(prompt,sentence):
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                    {"role": "user", "content": f"Translate the following text from English into Spanish using above context: {sentence}"}],
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





import mysql.connector

# Database configuration
db_config = {
    'host': '172.16.34.1',
    'port': 3307,
    'user': 'umls',
    'password': 'umls',
    'database': 'umls2024'
}

def get_synonyms1(keyword):
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Step 1: Find CUI for the given keyword
        cursor.execute("SELECT CUI FROM MRCONSO WHERE STR LIKE %s LIMIT 1", (f"%{keyword}%",))
        cuis = cursor.fetchall()

        if not cuis:
            print("No CUI found for the keyword.")
            return []

        synonyms = set()

        # Step 2: Retrieve synonyms for each CUI
        for cui in cuis:
            query_synonyms = """
            SELECT DISTINCT str FROM MRCONSO 
            WHERE cui = %s AND lat = 'ENG'
            """
            cursor.execute(query_synonyms, (cui[0],))
            synonyms.update([row[0] for row in cursor.fetchall()])

        cursor.close()
        conn.close()

        return list(synonyms)

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

def get_synonyms2(keyword):
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

import json
import os
from datetime import datetime

def save_to_json(file_path, text):
    data = {"text": text, "timestamp": datetime.now().isoformat()}
    
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"Warning: {file_path} contains invalid data. Overwriting the file.")
            existing_data = []
    else:
        existing_data = []

    # Append new entry
    existing_data.append(data)
    
    # Save back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4)