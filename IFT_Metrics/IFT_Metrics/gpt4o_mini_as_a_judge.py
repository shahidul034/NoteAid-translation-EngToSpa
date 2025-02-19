import requests
import json

# Replace with your actual API key and organization key
api_key = ""
org_key = ""


def get_openai_response(prompt, model):
    url = 'https://api.openai.com/v1/chat/completions'
    
    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'OpenAI-Organization': org_key
    }

    # Data payload
    data = {
        'messages': [
            {'role': 'system', 'content': prompt}
        ],
        'model': model,  # Replace with the chat model you want to use
        'temperature': 0.0
    }

    # Sending POST request to OpenAI API
    response = requests.post(url, headers=headers, json=data)

    # Handling response
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return ""


import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def llm_as_a_judge_prompt(source, translated):
    prompt = f"""### Instruction: Evaluate the quality of the translation related to healthcare.

### Scoring Criteria:

**Case 1: Clinical Usefulness**
- **5 points** if the translated clinical note is definitely useful and can be used in clinical settings.
- **4 points** if the translated clinical note is mostly useful and can be used in clinical settings.
- **3 points** if the translated clinical note is somewhat useful and can be used in clinical settings.
- **2 points** if the translated clinical note is not useful and cannot be used in clinical settings.
- **1 point** if the translated clinical note is not useful at all and cannot be used in clinical settings.

**Case 2: Detailed Evaluation**
- **Relevance and Coherence**: 0.25 points for clarity and grammatical accuracy.

- **Hallucinations**: 0.75 points for translation information solely from the English note, without external facts.

- **Accuracy**: 
    - **5 points** if the clinical note is clinically accurate for the given patient
    - **4 points** if the clinical note is mostly accurate for the given patient
    - **3 points** if the clinical note is somewhat accurate for the given patient
    - **2 points** if the clinical note is not accurate for the given patient
    - **1 point** if the clinical note is not accurate at all for the given patient

- **Overall Clarity**
    - **5 points** if the summary is clear and easy to understand
    - **4 points** if the summary is mostly clear and easy to understand
    - **3 points** if the summary is somewhat clear and easy to understand
    - **2 points** if the summary is not clear and easy to understand
    - **1 point** if the summary is not clear and easy to understand at all

- **Coverage**: percentage of the critical medical details mentioned in the original English clinical note are in the translated clinical note.
    - if the translated clinical note contains all of the medical details mentioned in the original clinical note, award **2 points**
    - else if the translated clinical note contains some of the critical medical details that is included in original clinical note, award points based on its coverage % out of 2 points
    - else **0 points**

### Input:
- **English clinical note:**: 
{source}

- **Spanish clinical note:**: 
{translated}

### Output:
- "score: <total points>"
- Briefly justify your score, up to 50 words.
"""
    return prompt


def get_sft_prompt(prompt):
    prompt += f"\n ### Response: \nscore: "
    # prompt += "\n ### Response: "
    return prompt

def get_score_raw(text):
    # Regular expression for matching integers and floats
    text = text.lower()
    pattern = "score:\s*(\d+).*(\d+)"
    
    # Searching for the pattern in the text
    nums = re.findall("\d+\.\d+", text[:15])
    if len(nums):
        match = nums[-1]
        return float(match)
    elif re.findall("\d+", text[:15]):
        return int(re.findall("\d+", text[:15])[-1])
    return -1

import re
def get_score(text):
    # Regular expression to find the pattern "Score: " followed by one or more digits
    text = text.lower()
    pattern = r"score:\s*(\d+)"

    # Using re.search to find the first occurrence of the pattern
    match = re.search(pattern, text)

    score = -1
    if match:
        # Extracting the matched group, which is the score
        score = match.group(1)
        try:
            score = float(score)
        except:
            score = -1
    return score

def get_final_score(text):
    val = get_score_raw(text)
    if val == -1:
        val = get_score(text)
    return val

def get_again_score(text, score):
    if score != -1: return score

    text = text.lower().split('score')[1][:10]
    val = get_score_raw(text)
    if val == -1:
        val = get_score(text)
    return val

def get_score_bucket(score):
    if score > 4: return 5
    if score > 3: return 4
    if score > 2: return 3
    if score > 1: return 2
    return 1

import time
def run_inference(data_path, summary_name, model='gpt-4o-mini'):
    
    test_data = pd.read_csv(data_path)
    test_data['judge_prompt'] = test_data.apply(lambda x: llm_as_a_judge_prompt(x['original_english'], x["translated_spanish"]), 1)
    
    all_responses = []
    all_scores = []
    for i in range(len(test_data)):
        if i%20 == 0:
            print(f"########################### {i} samples are processed")
            time.sleep(60)
        
        judge_prompt = test_data.iloc[i]['judge_prompt']
        response = get_openai_response(prompt=judge_prompt, model=model)
        all_responses.append(response)
        
        score = get_final_score(response)
        all_scores.append(score)
        
    return all_responses, all_scores