from starlette.responses import HTMLResponse
from fastapi import HTTPException, Request
from starlette.websockets import WebSocket
from langchain_openai_voice import OpenAIVoiceReactAgent
import json
from utils.audio_utils.websocket import websocket_stream
from utils.audio_utils.tools import TOOLS
from .router import router
from langsmith import traceable
import os 
from dotenv import load_dotenv
load_dotenv(override=True)
import uuid
from langsmith import Client
import time


client = Client()
# Load environment variables from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Get JSON data sent from client
    data = await websocket.receive_json()
    model = data.get("model_name")
    username = data.get("username")
    chat_history = data.get("chat_history")
    namespaces = data.get("namespaces")
    metafield = json.dumps(data.get("metafield"))
    system_prompt = data.get("system_prompt")
    session_id = websocket.headers.get("session_id", "")

    await websocket.send_json(data)

   
    # Setup instruction
    instruction = f"""{system_prompt}
    
    - Always use the tool to get context before answering.
    - Use the tool **every time** the user asks something.
    - Always include:
    - namespaces = {namespaces}
    - metafield = {metafield if metafield else ' '}

    If the fetched context isn’t relevant, answer using your own knowledge.
    Stop VOICE **immediately** if the user starts talking while you are answering."""

    agent = OpenAIVoiceReactAgent(
        model="gpt-4o-realtime-preview",
        tools=TOOLS,
        instructions=instruction
    )

    # Start tracing
    run_id = str(uuid.uuid4())
    start_time = time.time()

    browser_receive_stream = websocket_stream(websocket)
    await agent.aconnect(browser_receive_stream, websocket.send_text)

    # End tracing
    end_time = time.time()
    client.create_run(
        name="Audio_Stream",
        run_type="chain",
        inputs={
            "username": username,
            "session_id": session_id,
            "model": model,
            "namespaces": namespaces,
            "instruction": instruction
        },
        outputs={"status": "stream_sent"},
        metadata={"source": "WebSocket"},
        id=run_id,
        start_time=start_time,
        end_time=end_time,
        tags=["audio", "websocket"]
    )

  

@router.get("/audio")
async def homepage(request: Request):
    with open("frontend/static/index.html") as f:
        html = f.read()
        return HTMLResponse(html)


