from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"Adding {a} and {b}")
    
    return a + b

Tools = [add]