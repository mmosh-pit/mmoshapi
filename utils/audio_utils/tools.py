from langchain_core.tools import tool
from crud.stream import  get_relevant_context

@tool
async def generate_stream_tools(
    prompt: str,
    namespaces: list[str],
    metafield: str,

 ):
    """
    Retrieves relevant context for the user's query based on the given prompt, namespaces, and metafield.

    This tool should be called after informing the user that a response is being generated.

    Args:
        prompt (str): The user's input or question.
        namespaces (list[str]): List of context namespaces.
        metafield (str): Additional metadata for filtering context.

    Returns:
        str: Retrieved context or an error message if the process fails.
    """
    try:
    
        # Get relevant context 
        context = await get_relevant_context(prompt, namespaces, metafield )
        
        return context
    

    except Exception as processing_error:
        print(f"Error in message processing: {str(processing_error)}")
        error_msg = "I'm having trouble processing your request. Please try again."
        return error_msg 


TOOLS = [generate_stream_tools]  # Add other tools as needed
