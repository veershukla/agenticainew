from langchain_core.runnables import RunnableLambda
import graphviz

# --- Node 1: Ask for user's name ---
# Every function must receive one argument, even if it does not use it
def ask_name(_: dict) -> dict:
    name = input("Bot: What's your name? ")
    return {"name": name, "greeting": ""}

# --- Node 2: Greet the user ---
def greet_user(state: dict) -> dict:
    name = state.get("name") or "there"
    greeting = f"Hello {name}, nice to meet you!"
    print("Bot:", greeting)
    return {"name": name, "greeting": greeting}

# --- Build LCEL chain using | syntax ---
# The following will be executed in sequence like this: 
# output1 = ask_name(input)
# output2 = greet_user(output1)
chain = RunnableLambda(ask_name) | RunnableLambda(greet_user)

# --- Run the chain ---
if __name__ == "__main__":
    print("Starting conversation...")
    final_state = chain.invoke({})
    print("\nConversation complete!")
    print("Final State:", final_state)

    # --- Draw simple flow graph ---
    dot = graphviz.Digraph(comment="Greeting LCEL Flow")
    dot.node("A", "Ask Name")
    dot.node("B", "Greet User")
    dot.edge("A", "B")

'''    graph_path = "c://code//agenticai//greeting_lcel_graph.png"
    dot.render(graph_path, format="png", cleanup=True)
    print(f"Graph image saved at: {graph_path}")
'''