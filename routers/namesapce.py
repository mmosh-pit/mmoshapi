from .router import router
from fastapi import  HTTPException
from fastapi.responses import JSONResponse
import os
import random
from utils.variable_constant.pinecone_data import index_name
from pinecone import Pinecone
from langsmith import traceable

@router.get("/fetch_namespaces")
@traceable(name="fetch_namespaces")
async def fetch_namespaces():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize the index
        index = pc.Index(index_name)
        
        # Fetch index stats to get namespaces
        stats = index.describe_index_stats()
        namespaces = stats.namespaces

        # Initialize a list to store namespace data
        namespace_data = []

        # Process namespace information and fetch all unique metadata
        for namespace, ns_stats in namespaces.items():
            unique_metadata = set()
            if ns_stats.vector_count > 0:
                try:
                    # Query multiple vectors from the namespace
                    query_response = index.query(
                        vector=[random.random() for _ in range(768)],  # Random vector with 768 dimensions
                        top_k=min(ns_stats.vector_count, 100),  # Fetch up to 100 vectors or all if less
                        namespace=namespace,
                        include_metadata=True
                    )
                    for match in query_response.matches:
                        if match.metadata and 'custom_metadata' in match.metadata:
                            unique_metadata.add(match.metadata['custom_metadata'])
                except Exception as e:
                    print(f"Error fetching metadata for namespace {namespace}: {str(e)}")

            namespace_data.append({
                "namespace": namespace,
                "vector_count": ns_stats.vector_count,
                "unique_metadata": list(unique_metadata) if unique_metadata else ["No metadata available"]
            })

        return JSONResponse(status_code=200, content={"namespaces": namespace_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.delete("/delete_namespace")
@traceable(name="delete_namespace")
async def delete_namespace(namespace: str):
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize the index
        index = pc.Index(index_name)
        
        # Fetch index stats to get namespace information
        stats = index.describe_index_stats()
        namespaces = stats.namespaces

        if namespace not in namespaces:
            return JSONResponse(status_code=404, content={
                "message": f"Namespace '{namespace}' not found"
            })

        # Get the vector count before deletion
        vector_count_before = namespaces[namespace].get('vector_count', 0)

        # Delete all vectors in the namespace
        try:
            delete_response = index.delete(delete_all=True, namespace=namespace)
        except Exception as delete_error:
            return JSONResponse(status_code=500, content={
                "message": f"Error during deletion: {str(delete_error)}",
                "vector_count_before": vector_count_before
            })

        # Get the vector count after deletion
        stats_after = index.describe_index_stats()
        vector_count_after = stats_after.namespaces.get(namespace, {}).get('vector_count', 0)

        vectors_deleted = vector_count_before - vector_count_after

        return JSONResponse(status_code=200, content={
            "message": f"Attempted to delete all vectors in namespace '{namespace}'",
            "vectors_deleted": vectors_deleted,
            "vector_count_before": vector_count_before,
            "vector_count_after": vector_count_after
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": f"An error occurred: {str(e)}"
        })

