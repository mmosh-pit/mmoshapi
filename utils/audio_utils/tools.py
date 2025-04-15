from langchain_core.tools import tool
from routers.stream import generate_stream
from fastapi.responses import StreamingResponse
# from crud.stream import create_live_session, get_relevant_context
# from langchain_community.tools import TavilySearchResults


@tool
def add(a: int, b: int):
    """Add two numbers. Please let the user know that you're adding the numbers BEFORE you call the tool"""
    return a + b

@tool
def generate_stream_tool(
    model_name: str,
    prompt: str,
    username: str,
    chat_history: list[str],
    namespaces: list[str],
    metafield: str,
    system_prompt : str
 ):
    """Generate a stream of text from a given prompt and model."""
    return StreamingResponse(generate_stream(model_name,
    prompt,
    username,
    chat_history,
    namespaces,
    metafield,
    system_prompt), media_type="text/plain")

TOOLS = [generate_stream_tool]  # Add other tools as needed
