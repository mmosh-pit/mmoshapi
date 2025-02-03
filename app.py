import uvicorn
from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from utils.variable_constant.vertex_google import (project_id  ,location )
from routers import (metadata , namesapce , stream)
from middleware.largefile import LargeRequestMiddleware 
import vertexai
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('service_account.json')

import google.generativeai as genai
genai.configure(api_key=os.getenv('GENAI_API_KEY'))
# vertexai.init(project=project_id, location=location )
vertexai.init(project=project_id, location=location , credentials=credentials)


load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

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



app.include_router(metadata.router)
app.include_router(namesapce.router)
app.include_router(stream.router)




# Create the documents folder if it does not exist
if not os.path.exists('documents'):
    os.makedirs('documents')



# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
