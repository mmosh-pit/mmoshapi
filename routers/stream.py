from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse , Response
from .router import router
from crud.stream import generate_stream , generate
from crud.chat import get_chat_history
from langsmith import traceable
import os
import uuid


@router.post("/generate_stream/")
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
        model_name = data.get('model', '')
        session_id = request.headers.get('session_id', " ")

        # Validate required fields
        if not username or not prompt:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )
        # Define traced function dynamically
        @traceable(
            name="generate_stream",
            project_name=os.getenv('PROJECT_NAME'),
            metadata={
                "user_id": username,
                "session_id": session_id,
                "model_name": model_name,
            }
        )
        async def traced_generate_stream():
            return generate_stream(
                model_name=model_name,
                prompt=prompt,
                username=username,
                chat_history=chat_history,
                namespaces=namespaces,
                metafield=metafield,
                system_prompt=system_prompt,
            )

        # Return streaming response
        return StreamingResponse(
            await traced_generate_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/")
async def get_generate(request: Request) -> Response:
    """Endpoint handler for generation requests."""
    try:
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        chat_history = data.get('chat_history', [])
        namespaces = data.get('namespaces', [])
        metafield = data.get('metafield', '')
        system_prompt = data.get('system_prompt', '')
        model_name = data.get('model', '')
        session_id = request.headers.get('session_id', " ")


        if not username or not prompt:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )

        # Dynamically decorated inner function
        @traceable(
            name="generate",
            project_name=os.getenv('PROJECT_NAME'),
            metadata={
                "user_id": username,
                "session_id": session_id,
                "model_name": model_name,
            }
        )
        async def traced_generate():
            return await generate(
                model_name=model_name,
                prompt=prompt,
                username=username,
                chat_history=chat_history,
                namespaces=namespaces,
                metafield=metafield,
                system_prompt=system_prompt,
            )

        result = await traced_generate()
        return Response(result, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))