import re

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse


class PathNormalizationMiddleware(BaseHTTPMiddleware):
    """
    Redirects any path that contains a known valid segment
    (like /v1/, /models/ etc. ) after some extra unnecessary prefixes.
    """

    def __init__(self, app, valid_roots=None):
        super().__init__(app)
        # Valid entrypoints
        self.valid_roots = valid_roots or ["v1", "chat", "models", "embeddings"]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Check if path already starts with a valid root (to avoid redirect loops)
        path_starts_with_valid_root = any(path.startswith(f"/{root}") for root in self.valid_roots)
        if path_starts_with_valid_root:
            return await call_next(request)

        # Only redirect paths that contain a valid root but don't start with one
        # Priority: if path contains /v1/, redirect to /v1/... otherwise use the matched root
        if "/v1/" in path:
            # Extract everything after /v1/
            v1_pos = path.find("/v1/")
            v1_suffix = path[v1_pos + 4:]  # Skip "/v1/"
            new_path = f"/v1/{v1_suffix}"
            
            # Prevent redirect loops: if new path is same as old, don't redirect
            if new_path == path:
                return await call_next(request)
            
            query = request.url.query
            if query:
                new_path += f"?{query}"
            return RedirectResponse(url=new_path, status_code=301)  # Use 301 for permanent redirect

        # Otherwise, find the last valid root in the path
        pattern = r".*/(" + "|".join(map(re.escape, self.valid_roots)) + r")(/.*|$)"
        match = re.match(pattern, path)

        if match:
            matched_root = match.group(1)
            matched_suffix = match.group(2)
            new_path = f"/{matched_root}{matched_suffix}"
            
            # Prevent redirect loops: if new path is same as old, don't redirect
            if new_path == path:
                return await call_next(request)
            
            query = request.url.query
            if query:
                new_path += f"?{query}"
            return RedirectResponse(url=new_path, status_code=301)  # Use 301 for permanent redirect

        return await call_next(request)
