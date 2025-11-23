import gradio as gr
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# ---------------------------------
# State
# ---------------------------------
class ProductState(TypedDict):
    query: str
    results: str

# ---------------------------------
# ChromaDB setup
# ---------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)

# ---------------------------------
# Search Node
# ---------------------------------
def search_products(state: ProductState) -> ProductState:
    # Use similarity search with score
    results = vectordb.similarity_search_with_score(state["query"], k=3)
    
    if not results:
        state["results"] = "No products found"
        return state
    
    output = ["Search Results:"]
    
    for i, (doc, distance) in enumerate(results, 1):
        title = doc.metadata.get("title", "Unknown")
        content = doc.page_content[:150]
        # Cosine similarity: 1 - distance
        similarity = 1 - distance
        if similarity > 1: similarity = 1
        if similarity < -1: similarity = -1
        similarity_percent = round(similarity * 100, 1)
        output.append(f"{i} {title} Relevance {similarity_percent}%")
        output.append(f"{content}...")
    
    state["results"] = "\n".join(output)
    return state

# ---------------------------------
# Build LangGraph
# ---------------------------------
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.set_entry_point("search")
graph.add_edge("search", END)
app = graph.compile()

# ---------------------------------
# Gradio interface
# ---------------------------------
def chat_fn(message, history):
    result = app.invoke({"query": message})
    return result["results"]

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Product Search with LangGraph",
    examples=["wireless headphones", "laptop", "coffee maker"]
)

if __name__ == "__main__":
    demo.launch()
