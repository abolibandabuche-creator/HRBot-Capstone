# HRBot
A 24/7 HR Policy Assistant built with LangGraph and Streamlit. Employees can ask about leave, payroll, PF, WFH, and more — the bot answers from real policy documents and never makes things up.

## Features

- Ask questions about any HR policy in natural language
- Answers grounded in 10 real policy documents via RAG
- Self-reflection loop — bot scores its own answer and retries if it's not faithful enough
- Faithfulness score shown on every response so you can trust what it says
- Remembers your name and employee ID across the conversation
- Smart routing — knows when to search policy docs, fetch the date, or just chat
- Suggests common questions on the home screen
- Clean chat UI with source citations and metadata pills

## Tech Stack

- Python + Streamlit
- LangGraph (agentic pipeline)
- Groq API — LLaMA 3.3 70B
- ChromaDB (vector database)
- Sentence Transformers (embeddings)

## Getting Started

```bash
pip install -r requirements.txt
```

Set your Groq API key (free at console.groq.com):

```bash
# Windows
$env:GROQ_API_KEY="your_key_here"

# Mac / Linux
export GROQ_API_KEY="your_key_here"
```

Run the app:

```bash
streamlit run capstone_streamlit.py
```

## Files

- `agent.py` — LangGraph agent, all 8 nodes, build_agent() function
- `capstone_streamlit.py` — Streamlit chat UI
- `day13_capstone.py` — full pipeline with test suite and RAGAS evaluation
- `requirements.txt` — dependencies

## Policy Topics Covered

Annual Leave · Sick Leave · WFH · Payroll · Maternity/Paternity · Appraisal & PIP · Code of Conduct · PF & Gratuity · Resignation & FnF · Training & L&D

---

Aboli Bandabuche · Roll No: 2305996 · Agentic AI (2026)
