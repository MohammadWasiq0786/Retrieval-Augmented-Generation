import requests
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("EURI_API_KEY")

def generate_completion(prompt, model="gpt-4.1-nano", temperature=0.3):
    url = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": temperature
    }
    res = requests.post(url, headers=headers, json=payload)
    return res.json()['choices'][0]['message']['content']
