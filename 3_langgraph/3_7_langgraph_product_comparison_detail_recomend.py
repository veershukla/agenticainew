import os
import gradio as gr
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# Step 1: Setup embeddings + Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)

# Step 2: Define state and LLM
class State(TypedDict):
    query: str
    type: str
    result: str

llm = ChatOpenAI(
    model="gpt-4.1-mini",    # or gpt-4.1, gpt-4o-mini
    api_key=os.getenv("OPENAI_API_KEY")
)

# Step 3: Node functions
def classify(state: State) -> State:
    query = state["query"].lower()
    if "vs" in query or "compare" in query:
        state["type"] = "compare"
    elif "best" in query or "recommend" in query:
        state["type"] = "recommend"
    else:
        state["type"] = "specific"
    return state


def handle_specific(state: State) -> State:
    products = vectordb.similarity_search(state["query"], k=2)

    if not products:
        state["result"] = "No similar products found."
        return state

    titles = [doc.metadata.get("title", "Unknown product") for doc in products]
    
    prompt = f"Analyze this product:\n{titles[0]}"
    response = llm.invoke(prompt)

    state["result"] = response.content
    return state


def handle_compare(state: State) -> State:
    products = vectordb.similarity_search(state["query"], k=4)

    if not products:
        state["result"] = "No comparable products found."
        return state

    titles = [doc.metadata.get("title", "Unknown product") for doc in products]
    
    prompt = f"Compare these products:\n{titles}"
    response = llm.invoke(prompt)

    state["result"] = response.content
    return state


def handle_recommend(state: State) -> State:
    results = vectordb.similarity_search_with_score(state["query"], k=3)

    if not results:
        state["result"] = "No recommendations found."
        return state

    titles = []
    for doc, distance in results:
        title = doc.metadata.get("title", "Unknown product")
        cosine_similarity = 1 - distance
        titles.append(f"{title} (similarity: {cosine_similarity:.4f})")

    prompt = f"Recommend from these:\n{titles}"
    response = llm.invoke(prompt)

    state["result"] = response.content
    return state


# Step 4: Router
def route(state: State) -> str:
    return state["type"]

# Step 5: Build LangGraph
graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("specific", handle_specific)
graph.add_node("compare", handle_compare)
graph.add_node("recommend", handle_recommend)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route,
    {
        "specific": "specific",
        "compare": "compare",
        "recommend": "recommend"
    }
)

graph.add_edge("specific", END)
graph.add_edge("compare", END)
graph.add_edge("recommend", END)

runnable = graph.compile()

# Step 6: Gradio interface
def search(query, chat_history):
    result = runnable.invoke({"query": query})
    answer = f"Type: {result['type']}\n\n{result['result']}"
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

demo = gr.ChatInterface(
    fn=search,
    title="Conditional LangGraph (Chroma + OpenAI)",
    examples=["iPhone 15", "iPhone vs Samsung", "best laptop"],
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
