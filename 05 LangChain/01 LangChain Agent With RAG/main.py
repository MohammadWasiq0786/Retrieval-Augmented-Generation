import os
import json
import numpy as np
import requests
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA

EURI_API_KEY= "euri-......"
def euri_embed(text):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    response = requests.post(url, headers=headers, json=payload)
    embedding = np.array(response.json()['data'][0]['embedding'])
    return embedding

def euri_chat(messages, temperature=0.7, max_tokens=500):
    url = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
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
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation

class EuriLLM(LLM):
    def _call(self, prompt, stop=None, **kwargs) -> str:
        """Single prompt usage (e.g., LLMChain)"""
        return euri_chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

    def _generate(self, prompts, stop=None, **kwargs) -> LLMResult:
        """Batch prompt usage (e.g., Agents)"""
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

with open("data/founder_story.txt", "r", encoding="utf-8") as f:
    text = f.read()


chunks = [text[i:i+500] for i in range(0, len(text), 500)]
documents = [Document(page_content=chunk) for chunk in chunks]
    
from langchain.embeddings.base import Embeddings

class EuriEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [euri_embed(t).tolist() for t in texts]

    def embed_query(self, text):
        return euri_embed(text).tolist()

embedding_model = EuriEmbeddings()

faiss_index = FAISS.from_texts(
    texts=[doc.page_content for doc in documents],
    embedding=embedding_model
)

retriever = faiss_index.as_retriever()

def calculator_tool(query):
    """evaluate the arithmetic expression"""
    try:
        return str(eval(query))
    except Exception as e :
        return f"error : {e}"
    
def summarizer_tool(text):
    """Summarize any text"""
    return euri_chat([
        {"role": "system", "content": "You summarize content."},
        {"role": "user", "content": f"Summarize:\n{text}"}
    ])
    
import wikipedia

def wikipedia_tool(query):
    """Fetch a summary from Wikipedia."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Error: {e}"


def translate_tool(text_and_language):
    """
    Translate text to a target language.
    Example input: 'Hello World || French'
    """
    try:
        parts = text_and_language.split("||")
        text = parts[0].strip()
        target_language = parts[1].strip()
    except:
        return "Invalid input format. Use: Text || Language"

    prompt = [
        {"role": "system", "content": "You translate text."},
        {"role": "user", "content": f"Translate this into {target_language}:\n{text}"}
    ]
    return euri_chat(prompt)

def explain_code_tool(code):
    """Explain what this code does."""
    prompt = [
        {"role": "system", "content": "You are an expert programmer who explains code."},
        {"role": "user", "content": f"Explain what this code does:\n{code}"}
    ]
    return euri_chat(prompt)

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="MUST be used for ANY calculation request. Always use this tool to compute math expressions."

    ),
    Tool(
        name="Summarizer",
        func=summarizer_tool,
        description="Summarizes any text provided."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_tool,
        description="Searches Wikipedia and returns a summary. Input should be the search term."
    ),
    Tool(
        name="Translator",
        func=translate_tool,
        description="Translates text into a target language. Input format: 'Text || Language'."
    ),
    Tool(
        name="CodeExplainer",
        func=explain_code_tool,
        description="Explains what a code snippet does."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = EuriLLM()

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    # Use RetrievalQA first
    retrieved_answer = qa_chain({"query": user_input})["result"]

    # Let the agent decide whether to use tools/memory
    final_response = agent.invoke(f"{user_input}\nRetrieved Info: {retrieved_answer}")

    print("\n[DEBUG] Memory so far:")
    for m in memory.chat_memory.messages:
        print(f"{m.type.upper()}: {m.content}")
    
    print(f"\nBot: {final_response}\n")


