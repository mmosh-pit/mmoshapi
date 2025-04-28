from fastapi import APIRouter , HTTPException, Request
from fastapi.responses import Response
from .stream import generate_graph
from .stream_subgraph import generate_graph_2
from utils.relavent_state_grpah.tools import add  # import your custom tool functions
from .stream_react import generate_graph_3
router = APIRouter()

# @router.get("/")
# FastAPI endpoint using LangGraph
def build_state(data, session_id):
    return {
        "username": data.get("username"),
        "prompt": data.get("prompt"),
        "chat_history": data.get("chat_history", []),
        "namespaces": data.get("namespaces", []),
        "metafield": data.get("metafield", ""),
        "system_prompt": data.get("system_prompt", ""),
        "model_name": data.get("model", ""),
        "session_id": session_id
    }

@router.post("/generate_langgraph/")
async def get_generate(request: Request) -> Response:
    try:
        data = await request.json()
        session_id = request.headers.get("session_id", " ")

        if not data.get("username") or not data.get("prompt"):
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )

        input_state = build_state(data, session_id)
        final_state = await generate_graph.ainvoke(input_state)
        return Response(final_state["response"], media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# @router.post("/generate_langgraph_subgraph/")

# async def get_generate(request: Request) -> Response:
#     try:
#         data = await request.json()
#         session_id = request.headers.get("session_id", " ")

#         if not data.get("username") or not data.get("prompt"):
#             raise HTTPException(
#                 status_code=400,
#                 detail="Username and prompt fields are required"
#             )

#         input_state = build_state(data, session_id)
#         final_state = await generate_graph_2.ainvoke(input_state)
#         return Response(final_state["response"], media_type="text/plain")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/generate_langgraph_subgraph/")
async def generate_langgraph_subgraph(request: Request) -> Response:
   
    try:
        data = await request.json()
        session_id = request.headers.get("session_id", "")

        if not data.get("username") or not data.get("prompt"):
            raise HTTPException(
                status_code=400,
                detail="Username and prompt fields are required"
            )

        # Build the initial state (assumes build_state sets up base fields)
        input_state = build_state(data, session_id)
        # Attach your custom tool functions here
        # input_state["custom_tools"] = [add]

        # Invoke the compiled generate_graph pipeline
        final_state = await generate_graph_2.ainvoke(input_state)

        return Response(content=final_state.get("response", ""), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/generate_react_agent/")
async def generate_react_agent(request: Request) -> Response:
    try:
        data = await request.json()
        session_id = request.headers.get("session_id", "")

        # Validate required fields
        if not data.get("prompt") or not data.get("model"):
            raise HTTPException(
                status_code=400,
                detail="Both 'prompt' and 'model_name' are required"
            )

        # Build initial state for the graph
        input_state = build_state(data, session_id)

        # Invoke the LangGraph pipeline
        final_state = await generate_graph_3.ainvoke(input_state)
        return Response(
            content=final_state.get("response", ""),
            media_type="text/plain"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))