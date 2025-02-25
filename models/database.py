
from pydantic import BaseModel
import dns
from pymongo import MongoClient
import os
import certifi
ca_cert_path = certifi.where()

# MongoDB setup
# mongo_uri = os.getenv('MONGO_URI') + f'&tls=true&tlsCAFile={ca_cert_path}'
mongo_uri = os.getenv('MONGO_URI')

client = MongoClient(mongo_uri)
db = client.get_database("moral_panic_bot")
     
# Pydantic model for request validation
class GenerateRequest(BaseModel):
    username: str
    prompt: str
    namespaces: list[str] = []
    metafield: str = ""

    class Config:
        arbitrary_types_allowed = True