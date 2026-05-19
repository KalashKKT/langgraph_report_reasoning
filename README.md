# langgraph-grounded-doc-qa

LangGraph pipeline for **grounded question answering** over multi-page documents (e.g. financial reports). It reads documents page by page, extracts structured facts, synthesizes a cross-page answer, and validates that the final response is supported by extracted data.

## Features

- **Page-by-page analysis** — grounded summaries from a single page at a time
- **Structured extraction** — entities, metrics, dates, claims (Pydantic + LLM)
- **Cross-page reasoning** — builds a partial answer across pages
- **Answer validation** — flags unsupported or hallucinated claims
- **Conditional routing** — reads more pages or stops when sufficient / limits hit
- **Utilities** — PDF text extraction (PyMuPDF), table detection (Table Transformer)

## How it works

```
START → Page Analyzer → Data Extraction → Cross-Page Reasoning
                              ↓
                    [Sufficient?] ──no──→ Next Page → (loop)
                              ↓ yes
                    Query Resolution → Validation → END (or retry)
```

Each step is a **LangGraph node** sharing one `GraphState` (query, pages, extracted data, partial/final answer, reasoning trace).

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — prompts & LLM interface
- [Mistral AI](https://mistral.ai/) — `mistral-large-latest`
- [Pydantic](https://docs.pydantic.dev/) — structured outputs
- PyMuPDF, Transformers, pdf2image (utilities)

## Project structure

```
.
├── data_agents/
│   ├── langgraph_agent/
│   │   ├── graph.py              # Graph definition & run_document_graph()
│   │   ├── graph_execution.py    # Example run
│   │   └── requirements.txt
│   ├── reports/
│   │   ├── text_extraction/      # PDF → page strings (JSON)
│   │   └── table_detection.py    # Table Transformer cropping
│   └── stock_data/
│       └── ta.py                 # yfinance + SMC (standalone, optional)
└── .env.example
```

## Prerequisites

- Python 3.10+
- [Mistral API key](https://console.mistral.ai/)
- For table detection: Poppler (pdf2image), optional GPU

## Setup

```bash
git clone https://github.com/KalashKKT/langgraph-grounded-doc-qa.git
cd langgraph-grounded-doc-qa/data_agents/langgraph_agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `data_agents/.env` from the example:

```bash
cp ../.env.example ../.env
```

Edit `.env`:

```env
MISTRAL_API_KEY=your_key_here
```

> **Never commit `.env` or API keys.**

## Quick start

Run the sample pipeline (hardcoded Reliance-style pages + query):

```bash
cd data_agents/langgraph_agent
python graph_execution.py
```

Use your own document pages in code:

```python
from graph import system_graph, run_document_graph

pages = [
    "Page 0: Your first page text...",
    "Page 1: Your second page text...",
]

result = run_document_graph(
    pages=pages,
    user_query="What are the main risks mentioned in the report?",
    system_graph=system_graph,
    max_iterations=10,
)

print(result["final_answer"])
print(result["validation_status"])
```

## PDF ingestion (optional)

Extract pages from a PDF:

```bash
cd data_agents/reports/text_extraction
python text_extraction.py
```

Output: JSON list of `"Page N: ..."` strings — load into `pages` for `run_document_graph()`.

## Table detection (optional)

Detect and crop tables from PDF pages (`reports/table_detection.py`). Set `PDF_PATH`, `OUTPUT_FOLDER`, and Poppler path for your OS.

## Configuration

| Variable / param      | Description                          |
|-----------------------|--------------------------------------|
| `MISTRAL_API_KEY`     | Mistral API key                      |
| `max_iterations`      | Max graph loops (default: 10)        |
| `recursion_limit`     | LangGraph recursion limit (default: 100) |

## Design choices

- **Grounding** — analyzers and the final resolver are instructed to use only provided page/extracted data
- **Validation loop** — if the answer fails validation and pages remain, the graph can advance and retry
- **Not a multi-agent swarm** — one orchestrated graph with specialized nodes (analyzer, extractor, reasoner, validator), not separate autonomous agents

## Limitations

- Sample query in `graph_execution.py` may not match sample page content; replace both for real demos
- PDF/table/stock utilities are not fully wired into the LangGraph pipeline
- Sufficiency and validation rely on LLM judgment

