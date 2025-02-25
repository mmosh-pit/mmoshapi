PROJECT_ID = "mmoshbot"
REGION = "northamerica-northeast1"
BUCKET = "alis-new-bucket"
BUCKET_URI = f"gs://{BUCKET}"

# The number of dimensions for the textembedding-gecko@003 is 768
# If other embedder is used, the dimensions would probably need to change.
DIMENSIONS = 768

from google.cloud import aiplatform
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

my_index = aiplatform.MatchingEngineIndex(index_name='projects/879529711942/locations/northamerica-northeast1/indexes/6523323322757808128')

my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name='projects/879529711942/locations/northamerica-northeast1/indexEndpoints/2879770486726721536')

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

texts = [doc.page_content for doc in doc_splits]
metadatas = [doc.metadata for doc in doc_splits]

print(texts)

from langchain_google_vertexai import (
    VectorSearchVectorStore,
)

vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=REGION,
    gcs_bucket_name=BUCKET,
    index_id=my_index.name,
    endpoint_id=my_index_endpoint.name,
    embedding=embedding_model,
)

vector_store.add_texts(texts=texts, is_complete_overwrite=True)


