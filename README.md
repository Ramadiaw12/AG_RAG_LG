# 🧠 RAG-Powered Resume Analysis Agent

> An AI agent that reads, understands, and reasons over resumes using Retrieval-Augmented Generation — combining vector search with tool-augmented intelligence.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**CV Intelligence** is a production-grade AI pipeline that ingests PDF resumes, transforms them into a searchable vector knowledge base, and exposes an intelligent agent capable of answering natural language questions about candidates.

The system goes beyond simple keyword matching. By combining **RAG** (Retrieval-Augmented Generation) with a **tool-equipped LangChain agent**, it can correlate candidate skills with role requirements, retrieve company context, and synthesize insightful answers — all from a single conversational query.

This project demonstrates a complete, end-to-end AI engineering workflow: from raw document ingestion to a reasoning agent with external tool integration.

---

## ✨ Features

- 📄 **PDF Resume Ingestion** — Load one or multiple CVs from disk using LangChain document loaders
- ✂️ **Intelligent Text Chunking** — Token-aware splitting with `tiktoken` + `RecursiveCharacterTextSplitter` for semantically coherent chunks
- 🔢 **OpenAI Embeddings** — High-quality vector representations using `text-embedding-ada-002`
- 🗃️ **ChromaDB Vector Store** — Persistent, local vector database for fast similarity search
- 🔍 **Semantic Retriever** — Context-aware retrieval that surfaces the most relevant CV passages
- 🤖 **LangChain Agent** — GPT-4o-mini powered agent with reasoning and tool orchestration
- 🛠️ **Dual Tool System** — Resume retriever + company info tool for grounded, context-rich answers
- 💬 **Natural Language Interface** — Ask anything about the candidate in plain English

---

## 🏗️ Architecture

### Pipeline Overview

The system is composed of two distinct phases: the **Indexing Pipeline** (run once per CV) and the **Query Pipeline** (run at inference time).

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                        │
│                                                             │
│  PDF Resume  ──►  Document Loader  ──►  Text Splitter       │
│                                              │               │
│                                              ▼               │
│  ChromaDB  ◄──  Vector Store  ◄──  OpenAI Embeddings        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                             │
│  User Query  ──►  LangChain Agent  ──►  Tool Selection      │
│                         │                     │              │
│                         │         ┌───────────┴──────────┐  │
│                         │         │                       │  │
│                         ▼         ▼                       ▼  │
│                   retriever_tool         get_company_info    │
│                         │                       │            │
│                         └──────────┬────────────┘            │
│                                    ▼                          │
│                          Context Aggregation                  │
│                                    │                          │
│                                    ▼                          │
│                          GPT-4o-mini Synthesis                │
│                                    │                          │
│                                    ▼                          │
│                          Final Answer to User                 │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Flow — Step by Step

| Step | Component | Description |
|------|-----------|-------------|
| 1 | `PyPDFLoader` | Reads the raw PDF and extracts raw text per page |
| 2 | `RecursiveCharacterTextSplitter` | Splits text into overlapping chunks using `tiktoken` for token-precise boundaries |
| 3 | `OpenAIEmbeddings` | Converts each chunk into a high-dimensional vector |
| 4 | `ChromaDB` | Persists vectors locally for efficient k-NN similarity search |
| 5 | `Retriever` | At query time, embeds the user question and retrieves top-k matching chunks |
| 6 | `retriever_tool` | Wraps the retriever as a LangChain tool with a descriptive schema |
| 7 | `get_company_info` | External tool that provides simulated/enriched company context |
| 8 | `Agent (GPT-4o-mini)` | Reasons over retrieved context + tool outputs to generate a final answer |

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| LLM | OpenAI GPT-4o-mini | Reasoning, summarization, answer generation |
| Embeddings | OpenAI `text-embedding-ada-002` | Semantic vector representation |
| Orchestration | LangChain | Agent framework, tool binding, chain composition |
| Vector Store | ChromaDB | Local persistent vector database |
| Tokenizer | `tiktoken` | Token-aware text splitting |
| Document Loader | LangChain `PyPDFLoader` | PDF ingestion |
| Language | Python 3.10+ | Core implementation |

---

## 📁 Project Structure

```
cv-intelligence/
│
├── data/
│   └── resumes/                  # Drop PDF resumes here
│       └── candidate.pdf
│
├── vectorstore/
│   └── chroma_db/                # Persisted ChromaDB index (auto-generated)
│
├── src/
│   ├── ingestion.py              # PDF loading + chunking + embedding pipeline
│   ├── retriever.py              # ChromaDB retriever setup
│   ├── tools.py                  # LangChain tool definitions
│   ├── agent.py                  # Agent initialization and execution
│   └── utils.py                  # Helper functions
│
├── main.py                       # Entry point — run queries against the agent
├── .env                          # API keys (not committed)
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cv-intelligence.git
cd cv-intelligence

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```txt
langchain>=0.2.0
langchain-openai
langchain-community
langchain-chroma
chromadb
tiktoken
pypdf
python-dotenv
openai
```

---

## 🔐 Environment Variables

Create a `.env` file at the root of the project:

```bash
cp .env.example .env
```

Then fill in your credentials:

```env
# .env
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: customize ChromaDB path
CHROMA_PERSIST_DIR=./vectorstore/chroma_db
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## 🚀 Usage

### Step 1 — Index a Resume

```python
# src/ingestion.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import tiktoken

def index_resume(pdf_path: str, persist_dir: str = "./vectorstore/chroma_db"):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Token-aware chunking
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=lambda text: len(tokenizer.encode(text)),
    )
    chunks = splitter.split_documents(documents)

    # Embed and persist
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"✅ Indexed {len(chunks)} chunks from {pdf_path}")
    return vectorstore
```

### Step 2 — Query the Agent

```python
# main.py
from src.agent import create_agent

agent = create_agent()

response = agent.invoke({
    "input": "What are the candidate's main technical skills and do they match a backend engineering role at OpenAI?"
})

print(response["output"])
```

### Example Queries

```bash
python main.py --query "What degrees does the candidate hold?"
python main.py --query "How many years of experience does this candidate have with Python?"
python main.py --query "Does the candidate's profile fit a senior ML engineer role at DeepMind?"
```

---

## 🔧 Tools

The agent has access to two distinct tools that it selects and orchestrates autonomously:

### 1. `retriever_tool` — Resume Knowledge Base

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    name="resume_retriever",
    description=(
        "Use this tool to search the candidate's resume. "
        "Query it for: full name, education and degrees, work experience, "
        "technical skills, certifications, languages, and projects."
    ),
)
```

| Attribute | Value |
|-----------|-------|
| **Name** | `resume_retriever` |
| **Input** | Natural language query about the candidate |
| **Output** | Top-k most semantically relevant resume chunks |
| **Backend** | ChromaDB cosine similarity search |

---

### 2. `get_company_info` — Company Intelligence Tool

```python
from langchain.tools import tool

@tool
def get_company_info(company_name: str) -> str:
    """
    Returns structured information about a company.
    Use this to understand a company's tech stack, culture, or hiring needs
    before evaluating a candidate's fit.
    """
    company_data = {
        "openai": {
            "industry": "Artificial Intelligence",
            "tech_stack": ["Python", "PyTorch", "Kubernetes", "CUDA"],
            "culture": "Research-driven, fast-paced, high autonomy",
            "hiring_focus": "ML engineers, safety researchers, infrastructure engineers",
        },
        "google": {
            "industry": "Technology",
            "tech_stack": ["Go", "Python", "C++", "TensorFlow", "BigQuery"],
            "culture": "Data-driven, collaborative, large-scale systems",
            "hiring_focus": "SWE, SRE, ML engineers, PMs",
        },
    }
    key = company_name.lower().strip()
    return str(company_data.get(key, {"error": f"No data found for '{company_name}'"}))
```

| Attribute | Value |
|-----------|-------|
| **Name** | `get_company_info` |
| **Input** | Company name (string) |
| **Output** | Tech stack, culture, and hiring focus |
| **Backend** | Simulated data store (extensible to real APIs) |

---

## 🤖 Agent Behavior

The agent is initialized using LangChain's `create_openai_tools_agent` with GPT-4o-mini as the backbone:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert HR analyst and talent evaluator. "
        "You have access to a candidate's resume through the retriever tool, "
        "and you can fetch company data using the company info tool. "
        "Always ground your analysis in retrieved data. Be concise, professional, and precise."
    )),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm=llm, tools=[retriever_tool, get_company_info], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool, get_company_info], verbose=True)
```

**Decision Logic:**

```
User Query
    │
    ▼
Does this require resume data?  ──YES──►  Call retriever_tool
    │
    ▼
Does this require company data? ──YES──►  Call get_company_info
    │
    ▼
Synthesize all context  ──►  Generate final answer
```

The agent reasons step-by-step (ReAct-style), deciding which tools to call, in what order, and how to combine their outputs into a coherent, grounded response.

---

## 💡 Example Output

**Query:**
```
"Does this candidate qualify for a senior backend engineer role at Google?"
```

**Agent Trace:**
```
> Entering new AgentExecutor chain...
  Invoking: `resume_retriever` with "backend engineering experience Python distributed systems"
  → Retrieved 4 chunks: 5 years Python, microservices at Stripe, Kubernetes, gRPC...

  Invoking: `get_company_info` with "google"
  → Tech stack: Go, Python, C++, TensorFlow, BigQuery. Focus: SWE, SRE, ML engineers.

> Finished chain.
```

**Final Answer:**
```
The candidate presents a strong profile for a senior backend engineering role at Google.

Key strengths:
- 5+ years of Python experience aligns with Google's primary language stack
- Production experience with Kubernetes and distributed systems at Stripe
- Familiarity with gRPC matches Google's internal RPC infrastructure

Potential gaps:
- No demonstrated experience with Go, which is widely used at Google
- No exposure to BigQuery or GCP-native tooling

Verdict: Strong candidate — recommended for a technical screen. Advisable to assess
Go proficiency and cloud-native infrastructure knowledge during the interview process.
```

---

## 🔮 Future Improvements

- [ ] **Multi-CV Support** — Compare multiple candidates simultaneously across a single query
- [ ] **Real Company API** — Replace simulated company data with live integrations (LinkedIn, Glassdoor, Crunchbase)
- [ ] **Structured Output** — Return evaluation scores and match percentages as JSON for downstream use
- [ ] **Streamlit / FastAPI UI** — Add a web interface for HR teams with no coding background
- [ ] **Resume Anonymization** — PII stripping before indexing for privacy-compliant workflows
- [ ] **Multilingual Support** — Handle resumes in French, Spanish, Arabic, etc.
- [ ] **Evaluation Framework** — RAGAS-based benchmarking for retrieval quality and answer faithfulness
- [ ] **Memory** — Persistent conversation memory across sessions for iterative candidate assessment

---

## 🌍 Why This Project Matters

Traditional ATS (Applicant Tracking Systems) rely on brittle keyword matching. They miss context, fail on non-standard resume formats, and cannot reason about fit.

**CV Intelligence** demonstrates a fundamentally different approach:

- **Semantic understanding** over keyword matching
- **Reasoning** over a candidate's complete background, not isolated fields
- **Extensibility** — new tools (job boards, skills databases, salary APIs) can be plugged in without touching the core pipeline

This architecture is directly applicable to enterprise HR platforms, recruiting agencies, and any organization that needs to make faster, better-informed hiring decisions at scale.

---

## 👤 Author & Credits

**Developed by:** [DIAWANE Ramatoulaye](https://github.com/Ramadiaw12)

Built with:
- [LangChain](https://langchain.com) — Agent & chain orchestration
- [OpenAI](https://openai.com) — LLM & Embeddings
- [ChromaDB](https://trychroma.com) — Vector storage
- [tiktoken](https://github.com/openai/tiktoken) — Token-aware splitting

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with precision. Designed for scale. Powered by RAG.
</p>