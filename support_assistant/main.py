
from fastapi import FastAPI
from .models import AskRequest, AskResponse
from .graph import app_graph

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = app_graph.invoke({"query": request.query})
    return result["response"]
