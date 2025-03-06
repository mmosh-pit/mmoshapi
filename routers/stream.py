from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse , Response
from .router import router
from crud.stream import generate_stream 
from crud.chat import get_chat_history
from langsmith import traceable


@router.post("/generate_stream/")
@traceable(name="generate_stream")
async def get_generate_stream(request: Request) -> StreamingResponse:
    """Endpoint handler for streaming generation requests."""
    try:
        # Parse request data
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        chat_history = data.get('chat_history', [])
        namespaces = data.get('namespaces', [])
        metafield = data.get('metafield', '')
        system_prompt = data.get('system_prompt', '')

        # Validate required fields
        if not username or not prompt:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )
        

        
        
        # Return streaming response
        return StreamingResponse(
            generate_stream(
                prompt=prompt, 
                username=username,
                chat_history=chat_history,
                namespaces=namespaces,
                metafield=metafield,
                system_prompt=system_prompt,
                url = "stream"
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/generate/")
@traceable(name="generate")
async def get_generate_stream(request: Request) -> Response:
    """Endpoint handler for streaming generation requests."""
    try:
        # Parse request data
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        chat_history = data.get('chat_history', [])
        namespaces = data.get('namespaces', [])
        metafield = data.get('metafield', '')
        system_prompt = data.get('system_prompt', '')

        # Validate required fields
        if not username or not prompt:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )
        
        # Return streaming response
        return Response(
            generate_stream(
                prompt=prompt, 
                username=username,
                chat_history=chat_history,
                namespaces=namespaces,
                metafield=metafield,
                system_prompt=system_prompt,
                url = "generate"
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))