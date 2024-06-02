
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

from langchain_pinecone import PineconeVectorStore

index_name = "mmosh-index"

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

query = "How launchpad can be used?"
print(vectorstore.similarity_search(query))
