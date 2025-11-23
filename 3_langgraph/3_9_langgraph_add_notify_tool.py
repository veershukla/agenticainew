# ------------------------------------------------------------
# ReAct Agent using Chroma
# Includes:
# - Chroma Vector Store
# - SerpAPI for web search
# - ntfy for push notifications
# ------------------------------------------------------------

import os
import re
import requests
import gradio as gr
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic

# ---- Load environment variables ----
load_dotenv()

# ---- SETUP ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)

llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
)

# ---- TOOLS ----

def tool_search(query: str) -> str:
    """Searches local Chroma DB using cosine similarity (clean output)."""
    results = vectordb.similarity_search_with_score(query, k=2)
    if not results:
        return "No results found"
    
    output = []
    for doc, distance in results:
        title = doc.metadata.get("title", "Untitled")
        cosine_similarity = 1 - distance
        # Clamp between -1 and 1
        if cosine_similarity > 1:
            cosine_similarity = 1
        if cosine_similarity < -1:
            cosine_similarity = -1
        output.append(f"{title} similarity {cosine_similarity:.4f}")
    return "\n".join(output)


def tool_serp(query: str) -> str:
    """Performs web search using SerpAPI (clean output)."""
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": os.getenv("SERPAPI_API_KEY"), "num": 2}
    try:
        data = requests.get(url, params=params, timeout=10).json()
        organic = data.get("organic_results", [])
        if not organic:
            return "No SERP results found"
        results = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in organic[:2]]
        return "\n".join(results)
    except Exception as e:
        return f"SerpAPI error: {e}"


def tool_ntfy(message: str) -> str:
    """Sends a notification via ntfy.sh."""
    ntfy_topic = os.getenv("NTFY_TOPIC")
    ntfy_url = f"https://ntfy.sh/{ntfy_topic}"
    try:
        response = requests.post(ntfy_url, data=message, timeout=5)
        response.raise_for_status()
        return "Notification sent successfully"
    except requests.exceptions.RequestException as e:
        return f"Failed to send notification: {e}"


# ---- REACT AGENT ----
def react_agent(query: str, max_iterations=5):
    """ReAct-style reasoning loop with limited iterations."""
    history = []
    
    for iteration in range(max_iterations):
        history_text = "\n".join(history) if history else "No previous steps"
        
        prompt = f"""You are a ReAct agent. Follow this EXACT format:

Thought: [your reasoning in one sentence]
Action: [exactly one of: Search[query] | SerpSearch[query] | Ntfy[message] | Finalize[answer]]

Rules:
- Use Search[] for product database queries
- Use SerpSearch[] for web searches
- Use Ntfy[] to send notifications for iPhone-related queries
- Use Finalize[] ONLY when you have the complete answer
- Output ONLY Thought and Action, nothing else

Previous steps:
{history_text}

Original question: {query}

Your next step:"""

        try:
            response = llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error: {e}", history
        
        history.append(f"\nStep {iteration + 1}:\n{response}")
        
        action_match = re.search(r"Action\s*:\s*(\w+)\s*\[(.*?)\]", response, re.IGNORECASE | re.DOTALL)
        if not action_match:
            history.append("Error: Could not parse action. Stopping")
            break
        
        action = action_match.group(1).lower()
        arg = action_match.group(2).strip()
        
        if action == "search":
            observation = tool_search(arg)
            history.append(f"Observation:\n{observation}")
        elif action == "serpsearch":
            observation = tool_serp(arg)
            history.append(f"Observation:\n{observation}")
        elif action == "ntfy":
            observation = tool_ntfy(arg)
            history.append(f"Observation:\n{observation}")
        elif action == "finalize":
            return arg, history
        else:
            history.append(f"Unknown action: {action}")
            break
    
    history.append(f"Reached maximum iterations ({max_iterations})")
    return "Could not complete the task within the iteration limit", history


# ---- GRADIO UI ----
def respond(user_input, chat_history):
    if not user_input.strip():
        return chat_history
    
    chat_history.append((user_input, "Processing..."))
    yield chat_history
    
    final_answer, trace = react_agent(user_input)
    
    trace_text = "\n".join(trace)
    full_response = f"Answer:\n{final_answer}\n\nTrace:\n{trace_text}"
    
    chat_history[-1] = (user_input, full_response)
    yield chat_history


with gr.Blocks() as demo:
    gr.Markdown("# ReAct Agent with Claude + Chroma + SerpAPI + ntfy")
    gr.Markdown("Ask about products, search the web, or get notifications")
    
    chatbot = gr.Chatbot(label="Agent Conversation", height=500)
    
    with gr.Row():
        query = gr.Textbox(label="Your Question", placeholder="e.g., 'latest iPhone news' or 'wireless headphones'", scale=4)
        submit = gr.Button("Send", scale=1)
    
    gr.Examples(
        examples=[
            "Find wireless headphones",
            "Latest iPhone news",
            "Search for gaming laptops",
        ],
        inputs=query
    )
    
    submit.click(respond, [query, chatbot], [chatbot])
    query.submit(respond, [query, chatbot], [chatbot])

if __name__ == "__main__":
    demo.launch()
