from langchain_google_vertexai import VertexAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
loader = PyPDFLoader("documents/mmosh.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
doc_splits = text_splitter.split_documents(pages)

embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

from langchain_pinecone import PineconeVectorStore

index_name = "mmosh-index"

docsearch = PineconeVectorStore.from_documents(doc_splits, embeddings, index_name=index_name)

print(docsearch)