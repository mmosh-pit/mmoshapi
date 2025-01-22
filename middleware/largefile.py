
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi import  Request


class LargeRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == 'POST':
            MAX_BODY_SIZE = 100 * 1024 * 1024  # 100 MB
            content_length = int(request.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE:
                return JSONResponse(status_code=413, content={"message": "Request payload too large"})
        return await call_next(request)

# class ValidateUrlMiddleware(BaseHTTPMiddleware):


#     async def dispatch(self, request: Request, call_next):
#         if request.method == "POST" and request.url.path == "/upload":
#             form_data = await request.form()
#             urls = form_data.get("urls")

#             if urls and isinstance(urls, list) and len(urls) == 1 and urls[0] == 'None':
#                 urls = None
#                 form_data["urls"] = urls

#             request._body = form_data  # Update the request body with the modified form data

#             response = await call_next(request)
#             return response