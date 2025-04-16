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
    model: str,
    prompt: str,
    username: str,
    chat_history: list[str],
    namespaces: list[str],
    metafield: str,
    system_prompt : str
 ):
    """Generate a response using the given model, prompt, and context. 
        Please let the user know that you're generating the response BEFORE you call the tool.

        Parameters:
        - model_name: The name of the model to use.
        - prompt: The main input from the user.
        - username: The name of the user.
        - chat_history: List of previous chat messages.
        - namespaces: Related tags or domains for the chat.
        - metafield: Extra info related to the session.
        - system_prompt: Instructions for how the model should behave.  prompt: Instructions for how the model should behave.
        """
    print("Generating stream...")
    print("Model name:", model)
    print("Prompt:", prompt)
    print("Username:", username)
    print("Chat history:", chat_history)
    print("Namespaces:", namespaces)
    print("Metafield:", metafield)
    print("System prompt:", system_prompt)

    return StreamingResponse(generate_stream(model, prompt, username, chat_history, namespaces, metafield, system_prompt), media_type="text/event-stream")



TOOLS = [generate_stream_tool,  add]  # Add other tools as needed
