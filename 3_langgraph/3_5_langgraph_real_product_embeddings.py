# Dataset: https://www.kaggle.com/code/jayrdixit/amazon-product-dataset/input?select=amazon_products.csv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict
import pandas as pd
import gradio as gr
import chromadb
from chromadb.config import Settings

# --- Step 1: Define state ---
class ProductState(TypedDict):
    query: str
    results: str

# --- Step 2: Load dataset and embeddings ---
csv_path = "c://code//agenticai//3_langgraph//amazon_products.csv"
df = pd.read_csv(csv_path)
# df = df.head(1000)  # Limit for demo

texts = df["title"].astype(str).tolist()
metadatas = df.to_dict(orient="records")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Step 3: Setup Chroma (using L2 distance) ---
persist_dir = "c://code//agenticai//3_langgraph//chromadb"
collection_name = "products_collection"

client = chromadb.PersistentClient(path=persist_dir, settings=Settings())
existing_collections = [col.name for col in client.list_collections()]

if collection_name in existing_collections:
    print(f"Loading existing collection '{collection_name}'...")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
else:
    print(f"Creating new collection '{collection_name}'...")
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    vectordb.persist()

# --- Step 4: Define LangGraph nodes ---
def search_products(state: ProductState) -> ProductState:
    # Using cosine distance from Chroma (0 = identical)
    results = vectordb.similarity_search_with_score(state["query"], k=3)

    if not results:
        state["results"] = "No products found"
        return state

    output = []
    for i, (doc, distance) in enumerate(results, 1):
        title = doc.metadata.get("title", "Unknown Product")

        # Convert cosine distance -> cosine similarity
        cosine_similarity = 1 - distance

        output.append(
            f"{i}. {title} "
            f"(Cosine Distance: {distance:.4f}, "
            f"Cosine Similarity: {cosine_similarity:.4f})"
        )

    state["results"] = "\n".join(output)
    return state



# --- Step 5: Build LangGraph (simplified - removed format node) ---
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.set_entry_point("search")
graph.add_edge("search", END)
runnable = graph.compile()

# --- Step 6: Gradio handlers ---
def chat_fn(message, history):
    return runnable.invoke({"query": message})["results"]

# --- Step 7: Gradio UI ---
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Product Search (L2 Distance)",
    examples=["wireless headphones", "gaming laptop", "DSLR camera"],
)

if __name__ == "__main__":
    demo.launch()