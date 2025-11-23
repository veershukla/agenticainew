# pip install langchain-openai langchain-community langchain-huggingface langgraph chromadb requests gradio
import os
import gradio as gr
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# Step 1: Define state
class State(TypedDict):
    query: str
    vector: str
    serp: str
    llm: str

# Step 2: Setup embeddings, Chroma, and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",  # or gpt-4.1, gpt-4o, gpt-4o-mini
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)

# Step 3: Define graph nodes
def vector_search(state: State) -> State:
    """Search for similar products in Chroma vector database using cosine distance."""
    results = vectordb.similarity_search_with_score(state["query"], k=2)

    if not results:
        state["vector"] = "No matching products found."
        return state

    output = []
    for doc, distance in results:
        title = doc.metadata.get("title", "Unknown product")

        cosine_similarity = 1 - distance

        output.append(
            f"{title}\n"
            f"  • Cosine Distance: {distance:.4f}\n"
            f"  • Cosine Similarity: {cosine_similarity:.4f}"
        )

    state["vector"] = "\n\n".join(output)
    return state


def serp_search(state: State) -> State:
    """Fetch web results using SERP API"""
    url = "https://serpapi.com/search"
    params = {
        "q": f"{state['query']} reviews",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": 2
    }
    data = requests.get(url, params=params).json()
    organic = data.get("organic_results", [])

    if not organic:
        state["serp"] = "No web results found."
        return state

    results = [
        f"{r.get('title', 'No title')}: {r.get('snippet', '')}"
        for r in organic[:2]
    ]
    state["serp"] = "\n".join(results)
    return state


def llm_analyze(state: State) -> State:
    """Combine results and get AI-generated analysis using OpenAI"""
    prompt = (
        f"Analyze the following product search:\n\n"
        f"Query:\n{state['query']}\n\n"
        f"Vector DB Results:\n{state['vector']}\n\n"
        f"Web Results:\n{state['serp']}\n\n"
        f"Give a user-friendly summary and recommendation."
    )

    response = llm.invoke(prompt)
    state["llm"] = response.content
    return state

# Step 4: Build the graph
graph = StateGraph(State)
graph.add_node("vector_node", vector_search)
graph.add_node("serp_node", serp_search)
graph.add_node("llm_node", llm_analyze)

graph.set_entry_point("vector_node")
graph.add_edge("vector_node", "serp_node")
graph.add_edge("serp_node", "llm_node")
graph.add_edge("llm_node", END)

runnable = graph.compile()

# Step 5: Gradio interface
def search(query, chat_history):
    result = runnable.invoke({"query": query})
    answer = (
        f"Vector DB Results:\n{result['vector']}\n\n"
        f"Web Search Results:\n{result['serp']}\n\n"
        f"AI Analysis:\n{result['llm']}"
    )
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history


demo = gr.ChatInterface(
    fn=search,
    title="Product Search with LangGraph and Chroma (OpenAI Version)",
    examples=["iPhone 15", "best budget laptop", "Samsung Galaxy S23 reviews"],
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
