import gradio as gr
import pandas as pd
import sqlite3
from datetime import datetime
from typing import TypedDict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# ----------------------------
# State Definition
# ----------------------------
class ProductState(TypedDict):
    query: str
    results: str
    memory_hit: bool  # Added to indicate memory usage


# ----------------------------
# SQLite Memory Setup
# ----------------------------
conn = sqlite3.connect(r'c:/code/agenticai/3_langgraph/memory.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    query TEXT PRIMARY KEY,
    results TEXT,
    timestamp TEXT
)
""")
conn.commit()

def get_memory(query: str) -> Tuple[str, bool]:
    """Retrieve a cached answer if available"""
    cursor.execute("SELECT results, timestamp FROM memory WHERE query = ?", (query,))
    row = cursor.fetchone()
    if row:
        results, timestamp = row
        return f"(Loaded answer from memory, last updated: {timestamp})\n{results}", True
    return None, False

def save_memory(query: str, results: str):
    """Save new search result to SQLite memory"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT OR REPLACE INTO memory (query, results, timestamp)
    VALUES (?, ?, ?)
    """, (query, results, timestamp))
    conn.commit()


# ----------------------------
# ChromaDB Vector Search Setup
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)


# ----------------------------
# Node 1: Vector Search + Memory Check
# ----------------------------
def search_products(state: ProductState) -> ProductState:
    cached_result, memory_hit = get_memory(state["query"])
    if memory_hit:
        state["results"] = cached_result
        state["memory_hit"] = True
        return state

    # If not in memory → search in Chroma
    results = vectordb.similarity_search(state["query"], k=3)
    titles = [doc.metadata.get("title", "Unknown product") for doc in results]
    result_text = "\n".join([f"• {title}" for title in titles])
    state["results"] = result_text or "No products found"
    state["memory_hit"] = False

    # Save to memory
    save_memory(state["query"], result_text)
    return state


# ----------------------------
# Node 2: Format Response
# ----------------------------
def format_response(state: ProductState) -> ProductState:
    if state["results"]:
        prefix = "Found products:" if not state.get("memory_hit") else ""
        state["results"] = f"{prefix}\n{state['results']}" if prefix else state["results"]
    else:
        state["results"] = "No products found"
    return state


# ----------------------------
# LangGraph Pipeline
# ----------------------------
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
runnable = graph.compile()


# ----------------------------
# Search + Gradio Interface
# ----------------------------
def search(query):
    result = runnable.invoke({"query": query})
    return result["results"]

def chat_fn(message, history):
    return search(message)

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Product Search with Memory and ChromaDB",
    examples=["wireless headphones", "laptop", "coffee maker", "office chair"]
)

if __name__ == "__main__":
    demo.launch()
