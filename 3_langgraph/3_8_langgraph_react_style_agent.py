# ReAct-style agent using Chroma

import os
import re
import requests
import gradio as gr
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ---- SETUP ----
# Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)

# Initialize Anthropic Claude model
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    streaming=True,
)

# ---- TOOL 1: Local Vector Search ----
def tool_search(query: str) -> str:
    """
    Uses the Chroma vector DB to find the two most semantically similar documents
    to the user's query. Returns the titles of those documents along with
    cosine distance and cosine similarity.
    """
    # Use Chroma's similarity search with metadata including distance
    results = vectordb.similarity_search_with_score(query, k=2)

    if not results:
        return "No results found."

    output_lines = []
    for doc, distance in results:
        title = doc.metadata.get("title", "Untitled")

        cosine_distance = distance
        cosine_similarity = 1 - distance  # Convert to similarity

        output_lines.append(
            f"{title} "
            f"(Cosine Distance: {cosine_distance:.4f}, "
            f"Cosine Similarity: {cosine_similarity:.4f})"
        )

    return "\n".join(output_lines)


# ---- TOOL 2: SerpAPI Web Search ----
def tool_serp(query: str) -> str:
    """
    Calls SerpAPI for live web search results.
    """
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": os.getenv("SERPAPI_API_KEY"), "num": 2}

    try:
        data = requests.get(url, params=params).json()
        results = [f"{r['title']}: {r['snippet']}" for r in data.get("organic_results", [])[:2]]
        return "\n".join(results) if results else "No SERP results found."
    except Exception as e:
        return f"SerpAPI error: {e}"

# ---- REACT AGENT ----
def react_agent(query: str):
    """
    This is a ReAct-style reasoning loop.
    The LLM reasons step by step:
      - Thought: (decides what to do)
      - Action: (chooses Search, SerpSearch, or Finalize)
      - Observation: (receives results and continues reasoning)
    """
    state = {"query": query, "history": [f"User: {query}"], "final": ""}

    while True:
        # Construct prompt for the LLM
        prompt = f"""
You are a ReAct-style agent.
Always respond ONLY in this exact format:

Thought: (brief reasoning)
Action: (choose exactly one of)
  - Search[some query]
  - SerpSearch[some query]
  - Finalize[some final answer]

Conversation so far:
{chr(10).join(state['history'])}

User question: {state['query']}
Now continue.
"""

        # Stream response from Claude
        response = ""
        for chunk in llm.stream(prompt):
            if chunk.content:
                response += chunk.content
        response = response.strip()
        state["history"].append(response)

        # Example expected model output:
        # Thought: I should check vector DB for product info.
        # Action: Search[wireless earbuds with noise cancellation]

        # ---- Parse model action ----
        action_match = re.search(r"Action\s*:\s*(\w+)\s*\[(.*)\]", response)
        if not action_match:
            break  # if output doesn’t match format, stop

        action, arg = action_match.group(1).lower(), action_match.group(2).strip()

        # ---- Perform the chosen action ----
        if action == "search":
            obs = tool_search(arg)
            state["history"].append(f"Observation: {obs}")
        elif action == "serpsearch":
            obs = tool_serp(arg)
            state["history"].append(f"Observation: {obs}")
        elif action == "finalize":
            state["final"] = arg
            yield state["final"], "\n".join(state["history"])
            break

        # Stream intermediate trace
        yield None, "\n".join(state["history"])

# ---- GRADIO UI ----
with gr.Blocks() as demo:
    gr.Markdown("# ReAct Agent with Claude + Chroma + SerpAPI (Streaming)")

    chatbot = gr.Chatbot(label="Agent Trace")
    query = gr.Textbox(label="Ask something", placeholder="e.g. best air purifier under 10000")

    def respond(user_input, chat_history):
        chat_history.append(("User: " + user_input, ""))
        for final, trace in react_agent(user_input):
            if final:
                chat_history[-1] = (
                    chat_history[-1][0],
                    f"**Final Answer:** {final}\n\n---\n**Trace:**\n{trace}"
                )
                yield chat_history
            else:
                chat_history[-1] = (
                    chat_history[-1][0],
                    f"Working...\n\n**Trace so far:**\n{trace}"
                )
                yield chat_history

    query.submit(respond, [query, chatbot], [chatbot])

if __name__ == "__main__":
    demo.launch()
