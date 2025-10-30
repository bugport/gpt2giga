import base64
import json
import time
from typing import Optional

import httpx
import uuid


class TokenManager:
    """Fetches and caches a bearer token from a token endpoint using Basic auth.

    Expected behavior: POST to token_url with Authorization: Basic <b64>,
    content-type: application/x-www-form-urlencoded, body includes grant_type
    (default client_credentials) and optional scope. Response should contain
    access_token and expires_in seconds.
    """

    def __init__(
        self,
        token_url: Optional[str],
        basic_b64: Optional[str],
        grant_type: str = "client_credentials",
        scope: Optional[str] = None,
        timeout: int = 30,
    ):
        self.token_url = token_url
        self.basic_b64 = basic_b64
        self.grant_type = grant_type
        self.scope = scope
        self.timeout = timeout
        self._token: Optional[str] = None
        self._exp: float = 0.0

    def is_configured(self) -> bool:
        return bool(self.token_url and self.basic_b64)

    def get_token(self) -> Optional[str]:
        now = time.time()
        if self._token and now < (self._exp - 10):
            return self._token
        return self.refresh()

    def refresh(self) -> Optional[str]:
        if not self.is_configured():
            return None
        headers = {
            "Authorization": f"Basic {self.basic_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            # Some providers (e.g., GigaChat) require a unique request ID header
            "RqUID": str(uuid.uuid4()),
        }
        data = {}
        # Include grant_type only when specified (many endpoints accept scope-only)
        if self.grant_type:
            data["grant_type"] = self.grant_type
        if self.scope:
            data["scope"] = self.scope
        resp = httpx.post(
            self.token_url, data=data, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 1800))
        if token:
            self._token = token
            self._exp = time.time() + max(30, expires_in)
        return self._token


class TokenAwareClient:
    """Wraps an async client with retry-on-unauthorized using TokenManager.

    Expects the wrapped client to be a gigachat client with methods:
    - achat(chat)
    - astream(chat)
    - aembeddings(texts, model)
    - aget_models()
    - aget_model(model)

    On 401-like errors, refresh token and recreate the underlying client via
    a provided factory.
    """

    def __init__(self, manager: TokenManager, client_factory, initial_client):
        self._manager = manager
        self._client_factory = client_factory
        self._client = initial_client

    async def _retry(self, coro_factory):
        try:
            return await coro_factory(self._client)
        except Exception as e:
            # naive check for unauthorized; gigachat raises ResponseError with status
            msg = str(e)
            if "401" in msg or "Unauthorized" in msg or "invalid_token" in msg:
                if self._manager.is_configured():
                    self._manager.refresh()
                    self._client = self._client_factory()
                    return await coro_factory(self._client)
            raise

    async def achat(self, chat):
        return await self._retry(lambda c: c.achat(chat))

    async def astream(self, chat):
        async def gen():
            try:
                async for chunk in self._client.astream(chat):
                    yield chunk
            except Exception as e:
                msg = str(e)
                if "401" in msg or "Unauthorized" in msg or "invalid_token" in msg:
                    if self._manager.is_configured():
                        self._manager.refresh()
                        self._client = self._client_factory()
                        async for chunk in self._client.astream(chat):
                            yield chunk
                        return
                raise

        return gen()

    async def aembeddings(self, texts, model):
        return await self._retry(lambda c: c.aembeddings(texts=texts, model=model))

    async def aget_models(self):
        return await self._retry(lambda c: c.aget_models())

    async def aget_model(self, model: str):
        return await self._retry(lambda c: c.aget_model(model=model))


