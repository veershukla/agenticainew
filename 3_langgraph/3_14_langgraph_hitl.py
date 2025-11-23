import sqlite3
import requests
from bs4 import BeautifulSoup
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

# --------------------------------------------------------
# 1. State (must include a configurable key for checkpointing)
# --------------------------------------------------------
class WikiState(TypedDict):
    thread_id: str          # REQUIRED for checkpointing
    url: str
    page_text: str
    summary: str
    approved_summary: str
    hitl_required: bool


# --------------------------------------------------------
# 2. SQLite for memory
# --------------------------------------------------------
conn = sqlite3.connect(r"c:/code/agenticai/3_langgraph/hitl_wikipedia.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS wiki_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    summary TEXT,
    final TEXT
)
""")
conn.commit()

def save_to_db(url, summary, final):
    cursor.execute(
        "INSERT INTO wiki_approvals (url, summary, final) VALUES (?, ?, ?)",
        (url, summary, final)
    )
    conn.commit()


# --------------------------------------------------------
# 3. Helper: Scrape Wikipedia page text
# --------------------------------------------------------
def fetch_wikipedia_text(url: str) -> str:
    try:
        # Add User-Agent header to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Target the main content area more specifically
        content_div = soup.find("div", {"id": "mw-content-text"})
        if content_div:
            # Get paragraphs from the content div
            paragraphs = content_div.find_all("p")
            content = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        else:
            # Fallback to all paragraphs
            paragraphs = soup.select("p")
            content = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

        return content if content else "No readable content found on this page."
    except Exception as e:
        return f"Error fetching page: {e}"


# --------------------------------------------------------
# 4. LLM
# --------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")


# --------------------------------------------------------
# 5. Graph Nodes
# --------------------------------------------------------
def load_page(state: WikiState):
    text = fetch_wikipedia_text(state["url"])
    return {"page_text": text}


def generate_summary(state: WikiState):
    summary = llm.invoke([
        {"role": "user", "content": f"Summarize this Wikipedia text:\n\n{state['page_text']}"}
    ]).content

    return {
        "summary": summary,
        "hitl_required": True
    }


def wait_for_human(state: WikiState):
    # HITL: stop execution until user approves
    if state["hitl_required"]:
        return {"hitl_required": True}
    return state


def save_final(state: WikiState):
    save_to_db(state["url"], state["summary"], state["approved_summary"])
    return state


# --------------------------------------------------------
# 6. Build LangGraph (with checkpointing enabled)
# --------------------------------------------------------
graph = StateGraph(WikiState)

graph.add_node("load_page", load_page)
graph.add_node("summarize", generate_summary)
graph.add_node("hitl_wait", wait_for_human)
graph.add_node("save", save_final)

graph.set_entry_point("load_page")
graph.add_edge("load_page", "summarize")
graph.add_edge("summarize", "hitl_wait")
graph.add_edge("hitl_wait", "save")
graph.add_edge("save", END)

# FIXED: Remove config parameter from compile()
app = graph.compile(checkpointer=MemorySaver())


# --------------------------------------------------------
# 7. Gradio UI
# --------------------------------------------------------
def start_process(url):
    if not url.strip():
        return "Please provide a URL.", "", "", ""

    # Pass config when invoking, not when compiling
    state = app.invoke(
        {
            "thread_id": "session-1",
            "url": url,
            "page_text": "",
            "summary": "",
            "approved_summary": "",
            "hitl_required": False
        },
        config={"configurable": {"thread_id": "session-1"}}
    )

    return state["page_text"], state["summary"], state, state["summary"]


def approve(state, edited_summary):
    state["approved_summary"] = edited_summary
    state["hitl_required"] = False  # resume graph

    # Pass config when invoking
    final_state = app.invoke(
        state,
        config={"configurable": {"thread_id": "session-1"}}
    )
    return "Final summary saved!", edited_summary


# --------------------------------------------------------
# 8. UI Layout
# --------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## HITL Wikipedia Summary Generator (LangGraph)")

    url_box = gr.Textbox(
        label="Wikipedia URL",
        value="https://en.wikipedia.org/wiki/Geoffrey_Hinton",
        lines=1
    )

    run_btn = gr.Button("Fetch & Summarize Page")

    page_text_box = gr.Textbox(label="Wikipedia Extracted Text", lines=12)
    summary_box = gr.Textbox(label="LLM Summary", lines=6)
    edited_box = gr.Textbox(label="Edit Summary (HITL)", lines=6)

    approve_btn = gr.Button("Approve Summary")
    status_box = gr.Textbox(label="Status")

    state_box = gr.State()

    run_btn.click(start_process, inputs=[url_box], 
                  outputs=[page_text_box, summary_box, state_box, edited_box])

    approve_btn.click(approve, inputs=[state_box, edited_box], 
                      outputs=[status_box, edited_box])


if __name__ == "__main__":
    demo.launch()