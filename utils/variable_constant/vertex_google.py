import os

from dotenv import load_dotenv
load_dotenv()



project_id = os.getenv('VERTEX_PROJECT_ID')
model_id = "gemini-2.0-flash-exp"
location = "us-central1"

index_name = os.getenv('PINECONE_INDEX')