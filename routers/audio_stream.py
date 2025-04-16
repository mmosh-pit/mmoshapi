
from starlette.responses import HTMLResponse
from fastapi import HTTPException, Request
from starlette.websockets import WebSocket
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
import uvicorn
# from fastapi import WebSocket
from starlette.routing import Route, WebSocketRoute

from langchain_openai_voice import OpenAIVoiceReactAgent


from utils.audio_utils.websocket import websocket_stream
from utils.audio_utils.tools import TOOLS
from .router import router
import os 
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

data_tool = {}
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Get JSON data sent from client
    data = await websocket.receive_json()
    model_name = data.get("model_name")
    username = data.get("username")
    chat_history = data.get("chat_history")
    namespaces = data.get("namespaces")
    metafield = data.get("metafield")
    system_prompt = data.get("system_prompt")

    await websocket.send_json(data)

    browser_receive_stream = websocket_stream(websocket)

    agent = OpenAIVoiceReactAgent(
        model="gpt-4o-realtime-preview",
        tools=TOOLS,
        instructions=system_prompt,
    )

    await agent.aconnect(browser_receive_stream, websocket.send_text)


@router.get("/audio")
async def homepage(request: Request):
    with open("frontend/static/index.html") as f:
        html = f.read()
        return HTMLResponse(html)




