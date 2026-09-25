
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from .models import AskResponse
from .rag import retrieve
from .prompt import PROMPT_TEMPLATE

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list
    response: dict

def classify_intent(state: GraphState):
    query = state["query"].lower()
    intent = "policy_question" if any(k in query for k in POLICY_KEYWORDS) else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState):
    retrieved = retrieve(state["query"], k=3)
    top = retrieved[0] if retrieved else None
    if os.getenv("MOCK_LLM", "1") == "1":
        snippet = top["document"][:200] if top else "No matching policy context was found."
        answer = f"Based on the retrieved context: {snippet}"
        response = AskResponse(answer=answer, sources=[x["id"] for x in retrieved], confidence=1.0).model_dump()
    else:
        context = "\n\n".join(x["document"] for x in retrieved)
        prompt = PROMPT_TEMPLATE.format(question=state["query"], context=context)
        response = validate_with_retry(
            {"answer": f"LLM extension placeholder. Prompt prepared: {prompt[:100]}",
             "sources": [x["id"] for x in retrieved], "confidence": 0.8},
            [x["id"] for x in retrieved],
        )
    return {"retrieved": retrieved, "response": response}

def validate_with_retry(raw: dict, sources: list):
    """Validate structured output; retry up to 2 additional validation attempts."""
    last_error = None
    for _ in range(3):
        try:
            return AskResponse(**raw).model_dump()
        except Exception as exc:
            last_error = exc
            raw = {"answer": str(raw.get("answer", "")), "sources": sources, "confidence": 0.5}
    return {"answer": f"Structured output error: {last_error}", "sources": sources, "confidence": 0.0}

def direct_answer(state: GraphState):
    if os.getenv("MOCK_LLM", "1") == "1":
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = "Real-LLM extension is optional; no retrieval is used for general questions."
    return {"response": AskResponse(answer=answer, sources=[], confidence=1.0).model_dump()}

def route(state: GraphState):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route, {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    })
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()

app_graph = build_graph()
