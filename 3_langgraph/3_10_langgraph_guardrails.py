# ------------------------------------------------------------
# ReAct Agent using Claude + Chroma + SerpAPI + ntfy + Guardrails
# ------------------------------------------------------------

import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic

# ---- Load environment variables ----
load_dotenv()

# ------------------------------------------------------------
# SIMPLE GUARDRAILS
# ------------------------------------------------------------

# 1) Blocked words guardrail
BLOCKED_WORDS = ['hack', 'exploit', 'illegal', 'bomb', 'violence', 'malware', 'virus']

def has_blocked_words(text):
    text_lower = text.lower()
    for word in BLOCKED_WORDS:
        if word in text_lower:
            return True, f"Blocked word detected: {word}"
    return False, None


# 2) Max length guardrail
MAX_LENGTH = 120

def exceeds_length(text):
    if len(text) > MAX_LENGTH:
        return True, "Query too long. Please keep it brief."
    return False, None


# 3) Allow-list topics guardrail (simple check)
ALLOWED_TOPICS = ["phone", "iphone", "laptop", "product", "review", "camera", "tablet"]

def is_not_allowed_topic(text):
    text_lower = text.lower()
    if not any(word in text_lower for word in ALLOWED_TOPICS):
        return True, "Query not related to allowed topics (e.g. phones, laptops, products)."
    return False, None


# 4) Basic personal data guardrail (no regex, simple checks)
def contains_personal_info(text):
    if "@" in text and "." in text:
        return True, "Detected possible email address. Personal info not allowed."
    if any(char.isdigit() for char in text) and len(text) > 10:
        return True, "Detected possible phone/ID number. Not allowed."
    return False, None


# ------------------------------------------------------------
# Setup Vector DB
# ------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="c://code//agenticai//3_langgraph//chromadb",
    embedding_function=embeddings,
    collection_name="products_collection"
)

# ------------------------------------------------------------
# Setup LLM
# ------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    streaming=True,
)

# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
def tool_search(query: str) -> str:
    """
    Uses Chroma similarity_search_with_score() which returns cosine distance.
    Converts to cosine similarity using: similarity = 1 - distance.
    """
    results = vectordb.similarity_search_with_score(query, k=2)

    if not results:
        return "No results found."

    output_lines = []
    for doc, distance in results:
        title = doc.metadata.get("title", "Untitled")
        cosine_distance = distance
        cosine_similarity = 1 - distance  # convert

        output_lines.append(
            f"{title} "
            f"(Cosine Distance: {cosine_distance:.4f}, "
            f"Cosine Similarity: {cosine_similarity:.4f})"
        )

    return "\n".join(output_lines)



def tool_serp(query: str) -> str:
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": os.getenv("SERPAPI_API_KEY"), "num": 2}
    try:
        data = requests.get(url, params=params).json()
        results = [f"{r['title']}: {r['snippet']}" for r in data.get("organic_results", [])[:2]]
        return "\n".join(results) if results else "No SERP results found."
    except Exception as e:
        return f"SerpAPI error: {e}"


def tool_ntfy(message: str) -> str:
    ntfy_topic = os.getenv("NTFY_TOPIC")
    ntfy_url = f"https://ntfy.sh/{ntfy_topic}"
    try:
        response = requests.post(ntfy_url, data=message, timeout=5)
        response.raise_for_status()
        return "Notification sent successfully."
    except requests.exceptions.RequestException as e:
        return f"Failed to send notification: {e}"

# ------------------------------------------------------------
# Agent Logic with new guardrails
# ------------------------------------------------------------
def react_agent(query: str):

    # Run all simple guardrails
    for check in [has_blocked_words, exceeds_length, is_not_allowed_topic, contains_personal_info]:
        blocked, reason = check(query)
        if blocked:
            yield f"Cannot process query: {reason}", reason
            return

    state = {"query": query, "history": [f"User: {query}"], "final": ""}

    while True:
        prompt = f"""
You are a ReAct-style agent. 
You MUST always follow this exact output format:

Thought: (one short sentence of reasoning)
Action: (exactly one of the following)
- Search[some query]
- SerpSearch[some query]
- Ntfy[some message]
- Finalize[some final answer]

You should use the Ntfy tool to send a notification whenever the user asks 
about the 'latest iPhone' or related queries. After sending the notification,
continue to answer normally.

Do NOT output anything else. 
Do NOT answer directly unless using Finalize[].

Conversation so far:
{chr(10).join(state['history'])}

User question: {state['query']}
Now continue.
"""

        # Stream LLM output
        response = ""
        for chunk in llm.stream(prompt):
            if chunk.content:
                response += chunk.content
        response = response.strip()
        state["history"].append(response)

        # Parse action
        action_match = re.search(r"Action\s*:\s*(\w+)\s*\[(.*)\]", response)
        if not action_match:
            break

        action, arg = action_match.group(1).lower(), action_match.group(2).strip()

        if action == "search":
            obs = tool_search(arg)
            state["history"].append(f"Observation: {obs}")

        elif action == "serpsearch":
            obs = tool_serp(arg)
            state["history"].append(f"Observation: {obs}")

        elif action == "ntfy":
            obs = tool_ntfy(arg)
            state["history"].append(f"Observation: {obs}")

        elif action == "finalize":
            state["final"] = arg
            yield state["final"], "\n".join(state["history"])
            break

        yield None, "\n".join(state["history"])


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# ReAct Agent with Guardrails + Chroma + ntfy + SerpAPI")

    chatbot = gr.Chatbot(label="Agent Trace")
    query = gr.Textbox(label="Ask something", placeholder="e.g. latest iphone news")

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
