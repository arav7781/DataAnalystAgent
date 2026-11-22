# DataAnalystAgent

Upload a CSV and ask questions about it in plain English. A LangGraph agent decides which analysis to run,
executes it in Python, and returns the answer with charts.

## Architecture

```
Next.js chat UI (my-app/) ──▶ Flask API (backend/agent.py)
                                 ├─ POST /upload        store the CSV, return a dataset id
                                 ├─ POST /chat          run the agent on the question
                                 └─ GET  /serve_file/…  serve generated charts
                              LangGraph: tool-calling LLM ⇄ ToolNode (loops until no tool call)
```

Agent tools:

| Tool | What it does |
| --- | --- |
| `complete_python_task` | Runs generated pandas/NumPy code against the uploaded data |
| `automated_eda_tool` | Shape, types, missing values, distributions, correlations |
| `smart_visualization_tool` | Picks and renders suitable Plotly / seaborn charts |
| `ml_modeling_tool` | Trains and scores scikit-learn models (random forest classification/regression, clustering with K-Means + PCA) |
| `automated_pipeline` | Chains EDA, visualisation and modelling for a full first pass |

The LLM is Llama 3 (8B) on Groq, initialised with `init_chat_model`.

## Run locally

```bash
# backend
cd backend
pip install flask flask-cors pandas numpy plotly seaborn matplotlib scikit-learn langgraph langchain "langchain[groq]"
export GROQ_API_KEY=...
python agent.py

# frontend
cd my-app
npm install && npm run dev
```

## Tech

Python · Flask · LangGraph · pandas · scikit-learn · Plotly · Next.js · TypeScript
