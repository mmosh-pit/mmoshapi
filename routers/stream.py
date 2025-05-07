from fastapi import HTTPException, Request, Depends 
from fastapi.responses import StreamingResponse , Response
from .router import router
from crud.stream import generate_stream , generate
from crud.chat import get_chat_history
from langsmith import traceable
from middleware.auth import validate_token
from customtraceable import trace_with_dynamic_tags  # where you save trace_with_dynamic_tags


async def get_generate_stream_internal(
    request: Request,
    payload: dict,
) -> StreamingResponse:

    data = await request.json()
    username = data.get('username')
    prompt = data.get('prompt')
    chat_history = data.get('chat_history', [])
    namespaces = data.get('namespaces', [])
    metafield = data.get('metafield', '')
    system_prompt = data.get('system_prompt', '')
    model_name = data.get('model', '')

    return StreamingResponse(
        generate_stream(
            model_name=model_name,
            prompt=prompt,
            username=username,
            chat_history=chat_history,
            namespaces=namespaces,
            metafield=metafield,
            system_prompt=system_prompt,
        ),
        media_type="text/event-stream"
    )
 
@router.post("/generate_stream/")
async def get_generate_stream(request: Request, payload: dict = Depends(validate_token)) -> StreamingResponse:
    """Endpoint handler for streaming generation requests."""
    try:
        data = await request.json()
        username = data.get('username')
        if not username:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )


        traced_func = trace_with_dynamic_tags(username, "generate_stream")(get_generate_stream_internal)
        return await traced_func(request, payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/")
async def get_generate(request: Request, payload: dict = Depends(validate_token)) -> Response:
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

        if not username or not prompt:
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )

        # Get the complete response from the generate function
        result = await generate(
            model_name=model_name,
            prompt=prompt, 
            username=username,
            chat_history=chat_history,
            namespaces=namespaces,
            metafield=metafield,
            system_prompt=system_prompt,
        )
        return Response(result, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))