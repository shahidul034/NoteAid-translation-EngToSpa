import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from google import genai
import json
def gpucheck1():
    client = OpenAI(api_key=json.load(open('/home/mshahidul/project1/api.json', 'r'))['openai_api'])
    # client = genai.Client(api_key=json.load(open('/home/mshahidul/project1/api.json', 'r'))['gemini_api'])
    url = "http://sirchus.com/gpu_status.txt"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        text = response.text
        txt=f'''
            {text}

            Find the free available GPU (GPU Memory usage<=10%) using above context. Sort available GPU based on high configuration to low configuration. Show this table format.
            Add gpu information:
            GPU server 1 (omega):IP: 172.16.34.1
            GPU server 2 (alpha):IP: 172.16.34.21
            GPU server 3 (beta): IP: 172.16.34.22 
            GPU server 4 (gamma): IP: 172.16.34.29
            '''
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                        {"role": "user", "content": f"{txt}"}],
                temperature=0.5
            )
        ans=(response.choices[0].message.content)
        
        print(ans)
def gpucheck2():
    from google import genai
    client = genai.Client(api_key=json.load(open('/home/mshahidul/project1/api.json', 'r'))['gemini_api'])
    url = "http://sirchus.com/gpu_status.txt"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        text = response.text
        txt=f'''
            {text}

            Find the free available GPU (GPU Memory usage<=10%) using above context. Sort available GPU based on high configuration to low configuration. Show this table format.
            Add gpu information:
            GPU server 1 (omega):IP: 172.16.34.1
            GPU server 2 (alpha):IP: 172.16.34.21
            GPU server 3 (beta): IP: 172.16.34.22 
            GPU server 4 (gamma): IP: 172.16.34.29
            '''
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{txt}",
        )
        print(response.text)
gpucheck1()