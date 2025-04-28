# -*- coding: utf-8 -*-
# =====================
# Subgraph: get_relevant_context_graph
# =====================
from utils.library_calling.google_library import pinecone_store
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

class ContextState(TypedDict):
    prompt: str
    namespaces: List[str]
    metafield: str
    all_search_results: List  # intermediate results
    context: str
    error: str

# Node: extend namespaces
def set_search_namespaces(state: ContextState):
    state["namespaces"] += ["MMOSH"]
    return state

# Node: initialize results
def init_results(state: ContextState):
    state["all_search_results"] = []
    return state

# Node: search Pinecone
def call_search(state: ContextState):
    prompt = state["prompt"]
    metafield = state["metafield"]
    accum = []
    for ns in state["namespaces"]:
        try:
            batch = pinecone_store.similarity_search_with_score(
                prompt,
                namespace=ns,
                filter={"custom_metadata": metafield} if metafield and ns != "MMOSH" else None
            )
            for doc, score in batch:
                if score >= 0.6:
                    accum.append(doc)
        except Exception as e:
            state["error"] = str(e)
            return state
    state["all_search_results"] = accum
    return state

# Node: combine results into context
def combine_info(state: ContextState):
    state["context"] = " ".join(
        doc.page_content.strip() for doc in state["all_search_results"]
    )
    return state

# Node: error handler (subgraph)
def handle_error_sub(state: ContextState):
    print("Error retrieving context:", state.get("error", ""))
    state["context"] = ""
    return state

# Build subgraph
get_relevant_context_builder = StateGraph(ContextState)
get_relevant_context_builder.add_node("set_search_namespaces", set_search_namespaces)
get_relevant_context_builder.add_node("init_results", init_results)
get_relevant_context_builder.add_node("call_search", call_search)
get_relevant_context_builder.add_node("combine_info", combine_info)
get_relevant_context_builder.add_node("handle_error", handle_error_sub)
get_relevant_context_builder.set_entry_point("set_search_namespaces")
get_relevant_context_builder.add_edge("set_search_namespaces", "init_results")
get_relevant_context_builder.add_edge("init_results", "call_search")
get_relevant_context_builder.add_conditional_edges(
    "call_search",
    lambda state: "handle_error" if state.get("error") else "combine_info"
)
get_relevant_context_builder.add_edge("combine_info", END)
get_relevant_context_builder.add_edge("handle_error", END)
get_relevant_context_graph = get_relevant_context_builder.compile()

# =====================
# Parent Graph: generate_graph
# =====================
import os
from typing import List, Annotated, Sequence ,Callable
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from utils.variable_constant.vertex_google import project_id, location
from langchain.agents import Tool
from utils.relavent_state_grpah.tools import Tools
from utils.variable_constant.prompt import SYSTEM_PROMPT

class GenerateState(TypedDict):
    prompt: str
    model_name: str
    system_prompt: str
    chat_history: List[tuple[str, str]]
    namespaces: List[str]
    metafield: str
    context: str
    response: str
    error: str
    chat: object
    tools: List[Tool]
    agent_graph: object
    number_of_steps: int


model_name_groq = ["deepseek-r1-distill-llama-70b" , "deepseek-r1-distill-qwen-32b", "qwen-2.5-32b"  ,"qwen-qwq-32b" , "mistral-saba-24b",
                   "llama-3.2-1b-preview" , "llama-3.2-3b-preview","llama-3.3-70b-versatile","llama3-70b-8192","llama-3.1-8b-instant","llama3-8b-8192",
                  "mixtral-8x7b-32768","gemma2-9b-it",]

gemini_model_list = ["gemini-2.0-flash" , "gemini-2.0-flash-lite" , "gemini-2.0-pro-exp-02-05" , "gemini-1.5-flash" , "gemini-1.5-flash-8b" , "gemini-1.5-pro"]

anthropic_model_list = ["claude-3-7-sonnet-latest" , "claude-3-5-sonnet-latest" , 
                      "claude-3-5-sonnet-20240620" , "claude-3-opus-latest"]

open_ai_model_list  = [
    "gpt-4.5-preview",
    "gpt-4o",
    "chatgpt-4o-latest"
]

xai_model_list = ["grok-2-1212"]
# Node: create session (LLM)
async def create_live_session(state: GenerateState):
    """Create a new live session with appropriate configuration."""

    model_name = state["model_name"]
    if model_name in gemini_model_list:
        model_id = model_name
    
        # breakpoint()
        model = ChatGoogleGenerativeAI(
            google_api_key=os.getenv('GENAI_API_KEY'),
            model=model_id,  # Specify the appropriate Gemini model (e.g., text-bison)
            temperature=0,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
            project_id=project_id,  # Set your Google Cloud Project ID
            location=location  ,# Set your Google Cloud Region
        )
    
    elif model_name in open_ai_model_list:
        model = ChatOpenAI(
            model=model_name,  # e.g., "gpt-3.5-turbo" or "gpt-4"
            temperature=0,
            max_tokens=2048,
            timeout=None,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif model_name in anthropic_model_list:
        model = ChatAnthropic(
            model=model_name,  # e.g., "claude-v1"
            temperature=0,
            max_tokens=2048,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    
    elif model_name in model_name_groq:
        # Hypothetical integration for Grok / X AI
        model = ChatGroq(
            api_key=os.getenv("GROK_API_KEY"),
            model=model_name,
            temperature=0,
            max_tokens=2048
        )
    elif model_name in xai_model_list:
        model = ChatXAI(
            xai_api_key=os.getenv("XAI_API_KEY"),
            model=model_name,
        )
    else:
        state["error"] = f"Unsupported model: {model_name}"
        return "handle_error"

    state["chat"] = model
    return state


# Node: build React agent graph (inline tool wrapping)
def build_react_agent_graph(state: GenerateState):
    llm = state.get("chat")
    # wrap only known tools
    tools = [Tool(name=fn.__name__, func=fn, description=fn.__doc__ or "") for fn in Tools]
    system = SystemMessage(content=state.get("system_prompt") or SYSTEM_PROMPT)
    graph = create_react_agent(llm, tools=tools, prompt=system)
    state["agent_graph"] = graph

    state["used_tool"] = False
    return state

# Node: run React agent

def run_react_agent(state: GenerateState) -> GenerateState:
    graph = state['agent_graph']
    if not graph:
        state['error'] = 'Agent graph not initialized'
        return state
    last = None
    for step in graph.stream({'messages': state.get('chat_history', []) + [('user', state['prompt'])]},
                              stream_mode='values'):
        # if step.get('tool_calls'):
        #     state['used_tool'] = True

        print("Step", step)
        msg = step['messages'][-1]
        last = getattr(msg, 'content', None) or msg[1]
    state['response'] = last
    return state

# Node: run context subgraph as separate node
# This node references get_relevant_context_graph without invoking via run()
# langgraph will handle execution when wiring the subgraph


# Node: generate combined response using LLM
def generate_combined_response(state: GenerateState) -> GenerateState:
    # Prepare a prompt that merges tool output and retrieved context
    llm = state.get('chat')
    if not llm:
        state['error'] = 'LLM not initialized for final generation'
        return state
    response_text = state.get('response', '')
    context_text = state.get('context', '')
    system_msg = SystemMessage(content=state.get('system_prompt') or SYSTEM_PROMPT)
    user_msg = (f"""Here is an intermediate answer from a tool:
        {response_text}

        Here is the relevant context:
        "
                        f"Here is the relevant context:
        {context_text}

        "   
                "Please combine these into a final, comprehensive answer.""")
    try:
        final = llm.invoke([('system', system_msg), ('human', user_msg)])
        state['response'] = final.content
    except Exception as e:
        state['error'] = str(e)
    return state


def handle_error(state: GenerateState) -> GenerateState:
    """Global error handler for the generate_graph pipeline."""
    print("Error in pipeline:", state.get("error"))
    # Fallback response
    state["response"] = state.get("error", "Sorry, something went wrong.")
    return state

# Build parent graph
# Assemble parent graph
builder = StateGraph(GenerateState)
builder.set_entry_point('create_live_session')

# Core processing nodes
builder.add_node('create_live_session', create_live_session)
builder.add_node('build_react_agent_graph', build_react_agent_graph)
builder.add_node('run_react_agent', run_react_agent)

# Context subgraph as separate node
builder.add_node('get_relevant_context', get_relevant_context_graph)

builder.add_node('generate_combined_response', generate_combined_response)

# Error handler node
builder.add_node('handle_error', handle_error)

# Routing logic
builder.add_conditional_edges(
    'create_live_session',
    lambda s: 'handle_error' if s.get('error') else 'build_react_agent_graph'
)
builder.add_conditional_edges(
    'build_react_agent_graph',
    lambda s: 'handle_error' if s.get('error') else 'run_react_agent'
)
# Always proceed to context after agent
builder.add_edge('run_react_agent', 'get_relevant_context')
builder.add_conditional_edges(
    'get_relevant_context',
    lambda s: 'handle_error' if s.get('error') else 'generate_combined_response'
)
# After combination, end or error
builder.add_conditional_edges(
    'generate_combined_response',
    lambda s: 'handle_error' if s.get('error') else END
)
# Handle errors
builder.add_edge('handle_error', END)

# Compile the composite graph
generate_graph_2 = builder.compile()
