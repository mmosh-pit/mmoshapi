import os
from dotenv import load_dotenv
import requests

BACKEND_URL = os.getenv("NEXT_PUBLIC_BACKEND_URL", "http://localhost:6050")

def check_is_auth(endpoint: str, token: str):
    url = f"{BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        user = json_data.get("data", {}).get("user")

        if user is None:
            return None  # Invalid token

        return user  # Return user data
    except Exception as e:
        print(f"Auth check failed: {e}")
        return None
