import requests
import numpy as np
import wikipedia
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- EURI API KEY (replace with your own if needed) ---
EURI_API_KEY = "euri-....."
EURI_BASE_URL = "https://api.euron.one/api/v1/euri/alpha"

# --- EURI Embedding Function ---
def euri_embed(text):
    url = f"{EURI_BASE_URL}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        resp_json = response.json()
        if 'data' not in resp_json:
            print("[EURI EMBED ERROR] Response did not contain 'data':", resp_json)
            return np.zeros(1536)
        embedding = np.array(resp_json['data'][0]['embedding'])
        return embedding
    except Exception as e:
        print(f"[EURI EMBED ERROR] Exception: {e}")
        return np.zeros(1536)

# --- EURI Chat Completion Function ---
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

# --- Simple Tools ---
def calculator_tool(query):
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {e}"

def summarizer_tool(text):
    return euri_chat([
        {"role": "system", "content": "You summarize content."},
        {"role": "user", "content": f"Summarize:\n{text}"}
    ])

def wikipedia_tool(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception as e:
        return f"Error: {e}"

def rag_tool(query):
    """Answer questions about the founder story document using retrieval and LLM."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=EuriLLM(),
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.run(query)

# --- Embedding Model for FAISS ---
from langchain.embeddings.base import Embeddings
class EuriEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [euri_embed(t).tolist() for t in texts]
    def embed_query(self, text):
        return euri_embed(text).tolist()

# --- Load Document for Retrieval ---
with open("founder_story.txt", "r", encoding="utf-8") as f:
    text = f.read()
chunks = [text[i:i+500] for i in range(0, len(text), 500)]
documents = [Document(page_content=chunk) for chunk in chunks]
embedding_model = EuriEmbeddings()
faiss_index = FAISS.from_texts(
    texts=[doc.page_content for doc in documents],
    embedding=embedding_model
)
retriever = faiss_index.as_retriever()

# --- Custom LLM for LangChain ---
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

# --- Define Tools for Agents (now includes DocumentQA) ---
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for math calculations. Input should be an expression like '2+2'."
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
        name="DocumentQA",
        func=rag_tool,
        description=(
            "Only use this tool for questions about Sudhanshu, iNeuron, or the founder story. "
            "This tool searches the local founder_story.txt document for answers. "
            "For all other topics, use other tools or your own knowledge."
        )
    )
]

# --- Specialized Agents (Researcher, Teacher) ---
class SpecializedAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = EuriLLM()
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        self.system_prompt = system_prompt
    def run(self, user_input):
        # The agent LLM will decide if/when to use a tool
        try:
            return self.agent.run(f"{self.system_prompt}\n{user_input}")
        except Exception as e:
            return f"[Agent Error] {e}"

# --- Router Agent ---
class RouterAgent:
    def __init__(self, agents):
        self.llm = EuriLLM()
        self.agents = agents  # Dict: name -> SpecializedAgent
    def route(self, user_input):
        # Use LLM to decide which agent should handle the query
        system_prompt = (
            "You are a router agent. Given a user query, decide which agent should handle it: "
            "'Researcher' (for factual, research, or data questions) or 'Teacher' (for explanations, learning, or teaching). "
            "Reply with only the agent's name: Researcher or Teacher."
        )
        prompt = f"{system_prompt}\nUser query: {user_input}\nAgent:"
        agent_name = self.llm._call(prompt).strip().split()[0].capitalize()
        if agent_name not in self.agents:
            agent_name = "Teacher"  # Default fallback
        return agent_name
    def chat(self, user_input):
        agent_name = self.route(user_input)
        response = self.agents[agent_name].run(user_input)
        return f"[Router → {agent_name}]\n{response}"

# --- Create Specialized Agents ---
agents = {
    "Researcher": SpecializedAgent(
        name="Researcher",
        system_prompt="You are a research assistant. Be factual and concise. Use tools if needed. Use DocumentQA only for questions about Sudhanshu, iNeuron, or the founder story document. For other topics, use other tools or your own knowledge."
    ),
    "Teacher": SpecializedAgent(
        name="Teacher",
        system_prompt="You are a friendly teacher. Explain things simply. Use tools if needed. Use DocumentQA only for questions about Sudhanshu, iNeuron, or the founder story document. For other topics, use other tools or your own knowledge."
    )
}

# --- Create Router Agent ---
router = RouterAgent(agents)

# --- CLI Loop ---
print("\nConnected Multi-Agent Chat (type 'exit' to quit)")
print("Agents: Researcher, Teacher (chosen automatically by router)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"): break
    response = router.chat(user_input)
    print(f"{response}\n") 