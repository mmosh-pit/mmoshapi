from langsmith import traceable

def trace_with_dynamic_tags(username: str, method: str):
    def decorator(func):
        @traceable(name=method, tags=[f"{username}"])
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator