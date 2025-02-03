from .router import router
from fastapi import  HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import requests
import tempfile
from datetime import timedelta
from typing import List , Optional
import os

from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from utils.variable_constant.prompt import SYSTEM_PROMPT

from utils.variable_constant.pinecone_data import index_name


@router.post("/upload")
async def upload_file(
    name: str = Form(...),
    metadata: str = Form(...),
    file: UploadFile = File(None),
    urls : Optional[List[str]]  = Form(...),
    text : Optional[str]= Form(...),
    system_prompt : Optional[str] = Form(...)
):
    
    print("DEBUG -- name:", name)
    print("DEBUG -- metadata:", metadata)
    print("DEBUG -- raw urls:", urls)
    print("DEBUG -- text:", text)
    print("DEBUG -- system_prompt:", system_prompt)

    

    try:
        # 1) If no URLs at all, set it to None
        if not urls:
            urls = None

        # 2) If it's a list, remove placeholders
        elif isinstance(urls, list):
            # Remove any items that are empty or "None"
            cleaned_urls = []
            for u in urls:
                if u and u.strip().lower() != "none":
                    cleaned_urls.append(u.strip())

            if len(cleaned_urls) == 0:
                urls = None
            else:
                urls = cleaned_urls

        print("DEBUG -- final cleaned urls:", urls)

        # Now only proceed with actual URLs
        all_documents = []


       
        if urls:
            for url in urls:
                print("DEBUG -- calling requests.get() on url:", url)
                response = requests.get(url)
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type')
                    
                    if 'application/pdf' in content_type:
                        file_extension = 'pdf'
                    elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                        file_extension = 'docx'
                    else:
                        return JSONResponse(status_code=400, content={"message": f"URL {url} must point to a PDF or DOCX file."})
                    
                    # Save the content to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                        temp_file.write(response.content)
                        temp_file_path = temp_file.name
                    
                    if file_extension == 'pdf':
                        loader = PyPDFLoader(temp_file_path)
                    elif file_extension == 'docx':
                        loader = Docx2txtLoader(temp_file_path)
                    
                    documents = loader.load()
                    all_documents.extend(documents)
                    
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
                else:
                    return JSONResponse(status_code=402, content={"message": f"Failed to download file from {url}. Status code: {response.status_code}"})

        

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        # print(all_documents)
        doc_splits = text_splitter.split_documents(all_documents)
        # Split provided text, if any
        text_splits = []
        if text:
            with tempfile.NamedTemporaryFile(delete=False,  suffix=f'.txt') as temp_file:
                        temp_file.write(text.encode('utf-8'))
                        temp_text_file_path = temp_file.name
            text_splits = TextLoader(temp_text_file_path).load()
            # Clean up the temporary file
            print(temp_text_file_path)
            os.unlink(temp_text_file_path)

        for doc in doc_splits:
            doc.metadata['custom_metadata'] = metadata
            # Convert timedelta objects to strings
            for key, value in doc.metadata.items():
                if isinstance(value, timedelta):
                    doc.metadata[key] = str(value)
        
        docs = doc_splits + text_splits
        
        for doc in docs:
            doc.metadata['custom_metadata'] = metadata
            doc.metadata['system_prompt'] = system_prompt if system_prompt else SYSTEM_PROMPT
            # Convert timedelta objects to strings
            for key, value in doc.metadata.items():
                if isinstance(value, timedelta):
                    doc.metadata[key] = str(value)
        print(docs)
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

        PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name, namespace=name)
        
        return JSONResponse(status_code=200, content={"message": "Files uploaded successfully"})
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


@router.delete("/delete_by_metadata")
async def delete_by_metadata(metadata: str):
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize the index
        index = pc.Index(index_name)
        
        # Fetch index stats to get namespaces
        stats = index.describe_index_stats()
        namespaces = stats.namespaces

        total_deleted = 0
        debug_info = []

        for namespace in namespaces:
            try:
                # Query to find vectors with the specified metadata
                query_response = index.query(
                    vector=[0] * 768,  # Dummy vector
                    top_k=10000,  # Increase this if you have more vectors
                    namespace=namespace,
                    filter={"custom_metadata": {"$eq": metadata}},
                    include_metadata=True
                )
                
                matching_ids = [match.id for match in query_response.matches]
                matches_found = len(matching_ids)
                
                if matching_ids:
                    # Delete the matching vectors
                    delete_response = index.delete(ids=matching_ids, namespace=namespace)
                    
                    # Handle the case where delete_response is a dict
                    if isinstance(delete_response, dict):
                        deleted_count = delete_response.get('deleteCount', matches_found)
                    else:
                        deleted_count = getattr(delete_response, 'delete_count', matches_found)
                    
                    total_deleted += deleted_count
                    
                    debug_info.append(f"Namespace: {namespace}, Matches found: {matches_found}, Deleted: {deleted_count}")
                else:
                    debug_info.append(f"Namespace: {namespace}, No matches found")
                
                # Add information about the remaining metadata in this namespace
                post_delete_query = index.query(
                    vector=[0] * 768,
                    top_k=5,
                    namespace=namespace,
                    include_metadata=True
                )
                remaining_metadata = [match.metadata.get('custom_metadata') for match in post_delete_query.matches if match.metadata]
                debug_info.append(f"Remaining metadata samples in {namespace}: {remaining_metadata}")
            
            except Exception as e:
                debug_info.append(f"Error in namespace {namespace}: {str(e)}")

        if total_deleted > 0:
            return JSONResponse(status_code=200, content={
                "message": f"Deleted {total_deleted} vectors with metadata '{metadata}'",
                "debug_info": debug_info
            })
        else:
            return JSONResponse(status_code=404, content={
                "message": f"No vectors found or deleted with metadata '{metadata}'",
                "debug_info": debug_info
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

