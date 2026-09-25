
# Zepto Support Assistant

## Architecture
Ingestion -> Embedding -> Retrieval -> Generation

- `docs/doc_01.txt` ... `doc_08.txt`: exact Zepto policy corpus.
- `ingestion.py`: loads all 8 documents, uses `all-MiniLM-L6-v2`, and stores embeddings in ChromaDB.
- `rag.py`: embeds a query and retrieves the top 3 policy chunks using cosine similarity.
- `graph.py`: LangGraph `StateGraph` with `classify_intent`, `retrieve_and_answer`, and `direct_answer`, plus conditional routing.
- `prompt.py`: role-context-task-format-length prompt with a negative constraint and few-shot example.
- `models.py`: Pydantic request/response schemas.
- `main.py`: FastAPI `POST /ask`.

## Mock mode
`MOCK_LLM` is unset by default and therefore uses the required deterministic mock baseline.
Policy keywords route to retrieval. The retrieval node returns a canned answer beginning `Based on the retrieved context:`. General questions return a fixed policy-only message. Retrieval and ChromaDB run in both modes.

## Run
From the repository root:

```bash
pip install -r support_assistant/requirements.txt
python -m support_assistant.ingestion
uvicorn support_assistant.main:app --reload --port 8000
```

Example policy request:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{"query":"What is the delivery fee?"}"
```

Example general request:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{"query":"What is the capital of India?"}"
```

## Docker

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

Then call `POST http://127.0.0.1:7860/ask`.

## Output schema
Every response contains:
- `answer`: string
- `sources`: list of document/chunk IDs
- `confidence`: float from 0 to 1
