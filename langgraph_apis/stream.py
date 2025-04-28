
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from typing import AsyncGenerator
from utils.variable_constant.vertex_google import (project_id  ,location , model_id) 
from utils.relavent_state_grpah.pinecone_get import get_relevant_context_graph
import os
from utils.variable_constant.prompt import SYSTEM_PROMPT
from langsmith import traceable
import uuid
from langgraph.graph import StateGraph, END

from typing_extensions import TypedDict

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


class GenerateState(TypedDict):
    """State class for managing the generation process."""
 
    prompt : str
    model_name : str
    system_prompt : str
    chat_history : list[tuple[str, str]]
    namespaces : list[str]
    metafield : str
    chat : object
    context : str
    response : str
    error : str

@traceable(name="create_live_session" , run_type="tool" , project_name=os.getenv('PROJECT_NAME') , metadata={
    "user_id" : str(uuid.uuid4()),
    "session_id" : str(uuid.uuid4())
})

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

# Node: Build messages and invoke chat
def generate_response(state : GenerateState):
    """Generate a response using the chat model."""
    chat = state["chat"]
    prompt_used = state["system_prompt"] or SYSTEM_PROMPT
    context =  get_relevant_context_graph.invoke({
                "prompt": state["prompt"],
                "namespaces": state["namespaces"],
                "metafield": state["metafield"]
            })
    
    print("Context" , context)
    full_prompt = f"Context: {context}\n\nUser Question: {state['prompt']}\n\nAnswer:" if context else f"User Question: {state['prompt']}\n\nAnswer:"

    messages = [("system", prompt_used)] + list(map(tuple, state["chat_history"])) + [("human", full_prompt)]
    try:
        response = chat.invoke(messages)
        state["response"] = str(response.content)
    except Exception as e:
        state["error"] = str(e)
        return "handle_error"
    return state

# Define routing function
def route_after_response(state):
    if "error" in state:
        return "handle_error"
    return END


# Node: Handle error
def handle_error(state):
    print("Error:", state.get("error", "Unknown"))
    state["response"] = "Sorry, something went wrong."
    return state

# Build LangGraph
builder = StateGraph(GenerateState)

# Add all nodes
builder.add_node("create_session", create_live_session)
builder.add_node("generate_response", generate_response)
builder.add_node("handle_error", handle_error)

# Set entry point
builder.set_entry_point("create_session")

# Routing function after session creation
def route_after_session(state):
    return "handle_error" if "error" in state else "generate_response"


# Routing function after response generation
def route_after_response(state):
    return "handle_error" if "error" in state else END

# Use conditional routing functions
builder.add_conditional_edges("create_session", route_after_session)
builder.add_conditional_edges("generate_response", route_after_response)

# Error node leads to end
builder.add_edge("handle_error", END)

# Compile the graph
generate_graph = builder.compile()
