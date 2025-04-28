import uvicorn
from fastapi import FastAPI , Request
import os
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from utils.variable_constant.vertex_google import (project_id  ,location )
from routers import (chatmetadata , namesapce , stream , audio_stream)
from langgraph_apis.routers import router as langgraph_router
from middleware.largefile import LargeRequestMiddleware 
import vertexai
from fastapi.staticfiles import StaticFiles
from langsmith import traceable
from langsmith import Client
import uuid
# client = Client()

# from google.oauth2 import service_account

# credentials = service_account.Credentials.from_service_account_file('service_account.json')
# google_api_key = os.getenv('GOOGLE_API_KEY')
# print(os.getenv('GENAI_API_KEY'))

import google.generativeai as genai
genai.configure(api_key=os.getenv('GENAI_API_KEY'))
vertexai.init(project=project_id, location=location )
# vertexai.init(project=project_id, location=location , credentials=credentials)


load_dotenv(override=True)

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING')


app = FastAPI()

# Add the custom middleware
app.add_middleware(LargeRequestMiddleware)



# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



app.include_router(chatmetadata.router)
app.include_router(namesapce.router)
app.include_router(stream.router)
app.include_router(audio_stream.router)
app.include_router(langgraph_router)

app.mount("/audio", StaticFiles(directory="frontend/static"), name="static")

app.websocket_route("/ws")(audio_stream.websocket_endpoint)

# Create the documents folder if it does not exist
if not os.path.exists('documents'):
    os.makedirs('documents')



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
