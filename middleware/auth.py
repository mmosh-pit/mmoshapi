import os
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.database import db
import requests
from service.authservice import check_is_auth

# Define the HTTP Bearer auth scheme
security = HTTPBearer()

# Validation function
def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    user = check_is_auth("is-auth", token)

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # walletstr = user.get("address")

    # userwallet = collection.find_one({"wallet": walletstr})
    # if userwallet is None:
    #     raise HTTPException(status_code=401, detail="no fund in wallet")

    # if userwallet.get("value") < 1:
    #     raise HTTPException(status_code=401, detail="insufficent fund")


    return user