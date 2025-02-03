from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from .router import router
from crud.stream import generate_stream 
from crud.chat import get_chat_history

@router.post("/generate_stream/")
async def get_generate_stream(request: Request) -> StreamingResponse:
    """Endpoint handler for streaming generation requests."""
    try:
        # Parse request data
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        namespaces = data.get('namespaces', [])
        metafield = data.get('metafield', '')
        system_prompt = data.get('system_prompt', '')

        # Validate required fields
        if not username or not prompt:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )

        # Get chat history
        chat_history = await get_chat_history(username)
        
        # Return streaming response
        return StreamingResponse(
            generate_stream(
                prompt=prompt,
                username=username,
                chat_history=chat_history,
                namespaces=namespaces,
                metafield=metafield,
                system_prompt=system_prompt
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
