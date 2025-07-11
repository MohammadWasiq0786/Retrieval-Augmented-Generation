from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
import requests

# --- EURI API KEY ---
EURI_API_KEY = "euri-...."
EURI_BASE_URL = "https://api.euron.one/api/v1/euri/alpha"

# --- EURI LLM Wrapper ---
def euri_chat(messages, temperature=0.7, max_tokens=500):
    url = f"{EURI_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        resp_json = response.json()
        if 'choices' not in resp_json:
            print("[EURI CHAT ERROR] Response did not contain 'choices':", resp_json)
            if 'error' in resp_json:
                return f"[EURI CHAT ERROR] {resp_json['error'].get('message', 'Unknown error')}"
            return "[EURI CHAT ERROR] Unexpected response from EURI API."
        return resp_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"[EURI CHAT ERROR] Exception: {e}")
        return f"[EURI CHAT ERROR] Exception: {e}"

class EuriLLM(LLM):
    def _call(self, prompt, stop=None, **kwargs) -> str:
        return euri_chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
    def _generate(self, prompts, stop=None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)
    @property
    def _identifying_params(self):
        return {}
    @property
    def _llm_type(self):
        return "euri-llm"

# Prompt Template
prompt = PromptTemplate.from_template(
    "Classify the sentiment of this review as Positive, Negative, or Neutral:\n\n{text}"
)

# Output Parser
parser = StrOutputParser()

# EURI LLM
llm = EuriLLM()

# LCEL Chain
sentiment_chain = prompt | llm | parser

# Test
review = input("Enter a review: ")
result = sentiment_chain.invoke({"text": review})
print("Sentiment:", result) 