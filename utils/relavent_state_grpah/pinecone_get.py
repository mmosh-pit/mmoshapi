
from ..library_calling.google_library import pinecone_store 
from typing import List
import os, uuid
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    prompt: str
    namespaces: List[str]
    metafield: str
    all_search_results: List[str]
    relevant_info: str

# # Define the state keys used
# state = {
#     "prompt": str,
#     "namespaces": list,
#     "metafield": str,
#     "all_search_results": list,
#     "relevant_info": str
# }

# Node: set namespaces
def set_search_namespaces(state : State):
    state["namespaces"] += ["MMOSH"]
    return state

# Node: initialize results
def init_results(state):
    state["all_search_results"] = []
    return state

# Node: similarity search with score
def call_search(state: State):
    
    prompt = state["prompt"]
    metafield = state["metafield"]
    all_results = []

    for namespace in state["namespaces"]:
        try:
            results = pinecone_store.similarity_search_with_score(
                prompt,
                namespace=namespace,
                filter={"custom_metadata": metafield} if metafield and namespace != "MMOSH" else None
            )
            for doc, score in results:
                if score >= 0.6:
                    all_results.append(doc)
        except Exception as e:
            state["error"] = str(e)
            return "handle_error"

    state["all_search_results"] = all_results
    return state

# Node: combine info
def combine_info(state: State):
    state["relevant_info"] = " ".join([doc.page_content.strip() for doc in state["all_search_results"]])
    return state

# Node: print result
def print_info(state: State):
    print("Relevant info:", state["all_search_results"])
    return state

# Node: error handling
def handle_error(state: State):
    print("Error retrieving context:", state.get("error", "Unknown error"))
    state["relevant_info"] = ""
    return state

# Build the graph
graph = StateGraph(State)

graph.add_node("set_search_namespaces", set_search_namespaces)
graph.add_node("init_results", init_results)
graph.add_node("call_search", call_search)
graph.add_node("combine_info", combine_info)
graph.add_node("print_info", print_info)
graph.add_node("handle_error", handle_error)

graph.set_entry_point("set_search_namespaces")
graph.add_edge("set_search_namespaces", "init_results")
graph.add_edge("init_results", "call_search")
graph.add_conditional_edges("call_search", {
    "handle_error": handle_error,
    "default": combine_info
})
graph.add_edge("combine_info", "print_info")
graph.add_edge("print_info", END)
graph.add_edge("handle_error", END)

get_relevant_context_graph = graph.compile()


