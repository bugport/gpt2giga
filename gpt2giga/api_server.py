from contextlib import asynccontextmanager
import json
from pathlib import Path
import os
import httpx
import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from gigachat import GigaChat
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from gpt2giga.cli import load_config
from gpt2giga.logger import init_logger
from gpt2giga.middleware import PathNormalizationMiddleware
from gpt2giga.protocol import AttachmentProcessor, RequestTransformer, ResponseProcessor
from gpt2giga.router import router
from gpt2giga.auth import TokenManager, TokenAwareClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = getattr(app.state, "config", None)
    logger = getattr(app.state, "logger", None)

    if not config:
        from gpt2giga.cli import load_config
        from gpt2giga.logger import init_logger

        config = load_config()
        logger = init_logger(config.proxy_settings.log_level)

    app.state.config = config
    app.state.logger = logger
    # Optional token manager
    ps = config.proxy_settings
    token_manager = TokenManager(
        token_url=getattr(ps, "auth_token_url", None),
        basic_b64=getattr(ps, "auth_basic_b64", None),
        grant_type=getattr(ps, "auth_grant_type", "client_credentials"),
        scope=getattr(ps, "auth_scope", None),
    )
    # Insecure token fetch if requested
    if getattr(ps, "auth_insecure", False):
        token_manager.insecure = True

    # If configured, acquire token and inject into gigachat settings
    if token_manager.is_configured():
        token = token_manager.get_token()
        if token:
            config.gigachat_settings.access_token = token

    def build_client(endpoint_type: str = "default"):
        # Refresh token on each client build to ensure validity
        if token_manager.is_configured():
            token = token_manager.get_token()
            if token:
                config.gigachat_settings.access_token = token
        # Each endpoint gets its own GigaChat client with separate connection pool
        return GigaChat(**config.gigachat_settings.dict())

    # Shared upstream HTTP client (connection pooling) - for non-GigaChat direct requests
    def build_http_client(endpoint_type: str = "default"):
        try:
            http2 = str(os.getenv("GPT2GIGA_HTTP2", "false")).lower() == "true"
        except Exception:
            http2 = False
        # Endpoint-specific pool settings
        try:
            if endpoint_type == "embeddings":
                max_conns = int(os.getenv("GPT2GIGA_POOL_EMB_MAX_CONNECTIONS", os.getenv("GPT2GIGA_POOL_MAX_CONNECTIONS", "10") or 10) or 10)
                max_keep = int(os.getenv("GPT2GIGA_POOL_EMB_MAX_KEEPALIVE", os.getenv("GPT2GIGA_POOL_MAX_KEEPALIVE", "5") or 5) or 5)
            elif endpoint_type == "chat":
                max_conns = int(os.getenv("GPT2GIGA_POOL_CHAT_MAX_CONNECTIONS", os.getenv("GPT2GIGA_POOL_MAX_CONNECTIONS", "100") or 100) or 100)
                max_keep = int(os.getenv("GPT2GIGA_POOL_CHAT_MAX_KEEPALIVE", os.getenv("GPT2GIGA_POOL_MAX_KEEPALIVE", "20") or 20) or 20)
            else:
                max_conns = int(os.getenv("GPT2GIGA_POOL_MAX_CONNECTIONS", "100") or 100)
                max_keep = int(os.getenv("GPT2GIGA_POOL_MAX_KEEPALIVE", "20") or 20)
            keepalive_ms = int(os.getenv("GPT2GIGA_POOL_KEEPALIVE_TIMEOUT_MS", "60000") or 60000)
        except Exception:
            max_conns, max_keep, keepalive_ms = 100, 20, 60000

        limits = httpx.Limits(max_connections=max_conns, max_keepalive_connections=max_keep)
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=None)
        transport = httpx.AsyncHTTPTransport(
            retries=0,
            limits=limits,
        )
        return httpx.AsyncClient(
            http2=http2,
            transport=transport,
            timeout=timeout,
            headers={"Connection": "keep-alive"},
        )

    # Separate HTTP clients for different endpoints (if needed for direct requests)
    app.state.http_client = build_http_client("default")
    app.state.http_client_chat = build_http_client("chat")
    app.state.http_client_embeddings = build_http_client("embeddings")

    # Expose client factories and token manager for runtime rebuilds
    app.state.build_client = build_client
    app.state.token_manager = token_manager
    app.state.build_http_client = build_http_client

    # Separate GigaChat clients for each endpoint (each maintains its own connection pool)
    app.state.gigachat_client_chat = build_client("chat")
    app.state.gigachat_client_embeddings = build_client("embeddings")
    # Default client for backward compatibility (used by /models endpoints)
    app.state.gigachat_client = app.state.gigachat_client_chat
    
    # Separate TokenAwareClient wrappers for each endpoint
    app.state.client_chat = TokenAwareClient(
        token_manager, 
        lambda: build_client("chat"), 
        app.state.gigachat_client_chat
    )
    app.state.client_embeddings = TokenAwareClient(
        token_manager,
        lambda: build_client("embeddings"),
        app.state.gigachat_client_embeddings
    )
    # Default client for backward compatibility
    app.state.client = app.state.client_chat

    attachment_processor = AttachmentProcessor(app.state.gigachat_client)
    app.state.request_transformer = RequestTransformer(config, attachment_processor)
    app.state.response_processor = ResponseProcessor()

    # Load adaptivity and timeout state if present
    try:
        cfg_dir = Path(__file__).resolve().parent / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        adapt_path = cfg_dir / "adaptivity.json"
        if adapt_path.exists():
            with adapt_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                app.state._emb_rate = state
        # Load timeout state (e.g., stable median)
        timeout_path = cfg_dir / "timeout.json"
        if timeout_path.exists():
            with timeout_path.open("r", encoding="utf-8") as f:
                tout = json.load(f)
            if isinstance(tout, dict):
                # merge into emb_rate for shared median use
                rate = getattr(app.state, "_emb_rate", {}) or {}
                sm = tout.get("stable_med_ms")
                if isinstance(sm, (int, float)):
                    rate.setdefault("samples", [])
                    rate["stable_med_ms"] = float(sm)
                    app.state._emb_rate = rate
    except Exception:
        pass

    # Initialize idle watcher state
    app.state.last_activity = app.state.last_activity if hasattr(app.state, "last_activity") else 0.0
    app.state._pool_restart_idle_ms = 0  # Track when we last restarted pool
    async def _idle_watcher(app):
        try:
            warn_ms = int(os.getenv("IDLE_WARN_MS", "30000") or 30000)
            tick_ms = int(os.getenv("IDLE_CHECK_MS", "5000") or 5000)
            close_pool_ms = int(os.getenv("IDLE_CLOSE_POOL_MS", "60000") or 60000)
        except Exception:
            warn_ms, tick_ms, close_pool_ms = 30000, 5000, 60000
        while True:
            try:
                now = asyncio.get_event_loop().time()
                app.state.last_idle_ms = 0
                if getattr(app.state, "last_activity", 0) > 0:
                    idle_ms = int((now - app.state.last_activity) * 1000.0)
                    app.state.last_idle_ms = idle_ms
                    logger = getattr(app.state, "logger", None)
                    if idle_ms >= warn_ms and logger:
                        logger.info("Idle for %d ms", idle_ms)
                    # Close and restart pool when idle threshold exceeded
                    if idle_ms >= close_pool_ms:
                        last_restart = getattr(app.state, "_pool_restart_idle_ms", 0)
                        # Only restart once per idle period
                        if idle_ms - last_restart >= close_pool_ms:
                            try:
                                if logger:
                                    logger.info("Idle timeout %d ms: closing connections and restarting pool", idle_ms)
                                # Close and rebuild endpoint-specific HTTP clients
                                build_http = getattr(app.state, "build_http_client", None)
                                build = getattr(app.state, "build_client", None)
                                tman = getattr(app.state, "token_manager", None)
                                
                                # Chat client
                                http_client_chat = getattr(app.state, "http_client_chat", None)
                                if http_client_chat:
                                    try:
                                        await http_client_chat.aclose()
                                    except Exception:
                                        pass
                                if build_http:
                                    app.state.http_client_chat = build_http("chat")
                                if build and tman:
                                    new_gc_chat = build("chat")
                                    app.state.gigachat_client_chat = new_gc_chat
                                    app.state.client_chat = TokenAwareClient(tman, lambda: build("chat"), new_gc_chat)
                                
                                # Embeddings client
                                http_client_emb = getattr(app.state, "http_client_embeddings", None)
                                if http_client_emb:
                                    try:
                                        await http_client_emb.aclose()
                                    except Exception:
                                        pass
                                if build_http:
                                    app.state.http_client_embeddings = build_http("embeddings")
                                if build and tman:
                                    new_gc_emb = build("embeddings")
                                    app.state.gigachat_client_embeddings = new_gc_emb
                                    app.state.client_embeddings = TokenAwareClient(tman, lambda: build("embeddings"), new_gc_emb)
                                
                                # Update default client for backward compatibility
                                app.state.gigachat_client = app.state.gigachat_client_chat
                                app.state.client = app.state.client_chat
                                app.state._pool_restart_idle_ms = idle_ms
                                if logger:
                                    logger.info("Pool restarted after idle timeout")
                            except Exception as e:
                                if logger:
                                    logger.warning("Failed to restart pool on idle timeout: %s", e)
                await asyncio.sleep(max(0.001, tick_ms / 1000.0))
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)

    watcher_task = asyncio.create_task(_idle_watcher(app))

    yield

    # Save adaptivity and timeout state on shutdown
    try:
        rate = getattr(app.state, "_emb_rate", None)
        if isinstance(rate, dict):
            cfg_dir = Path(__file__).resolve().parent / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            adapt_path = cfg_dir / "adaptivity.json"
            tmp = adapt_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(rate, f, ensure_ascii=False, indent=2)
            tmp.replace(adapt_path)
            # timeout.json: persist a minimal subset
            timeout_path = cfg_dir / "timeout.json"
            tout_tmp = timeout_path.with_suffix(".tmp")
            payload = {"stable_med_ms": float(rate.get("stable_med_ms", 0.0) or 0.0)}
            with tout_tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tout_tmp.replace(timeout_path)
    except Exception:
        pass

    # Close endpoint-specific HTTP clients
    try:
        http_client_chat = getattr(app.state, "http_client_chat", None)
        if http_client_chat:
            await http_client_chat.aclose()
        http_client_emb = getattr(app.state, "http_client_embeddings", None)
        if http_client_emb:
            await http_client_emb.aclose()
        # Fallback for backward compatibility
        http_client = getattr(app.state, "http_client", None)
        if http_client:
            await http_client.aclose()
    except Exception:
        pass

    # Stop idle watcher
    try:
        watcher_task.cancel()
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan, title="Gpt2Giga converter proxy")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # /some_prefix/another_prefix/v1/... -> /v1/...
    # /api/v1/embeddings -> /v1/embeddings/
    app.add_middleware(
        PathNormalizationMiddleware,
        valid_roots=["v1", "chat", "models", "embeddings", "responses"],
    )

    @app.get("/", include_in_schema=False)
    async def docs_redirect():
        return RedirectResponse(url="/docs")

    app.include_router(router)
    app.include_router(router, prefix="/v1", tags=["V1"])
    return app


def run():
    config = load_config()
    proxy_settings = config.proxy_settings
    logger = init_logger(proxy_settings.log_level)

    app = create_app()
    app.state.config = config
    app.state.logger = logger

    logger.info("Starting Gpt2Giga proxy server...")
    logger.debug(f"Proxy settings: {proxy_settings}")
    logger.debug(
        f"GigaChat settings: {config.gigachat_settings.dict(exclude={'password', 'credentials', 'access_token'})}"
    )
    uvicorn.run(
        app,
        host=proxy_settings.host,
        port=proxy_settings.port,
        log_level=proxy_settings.log_level.lower(),
        ssl_keyfile=proxy_settings.https_key_file if proxy_settings.use_https else None,
        ssl_certfile=(
            proxy_settings.https_cert_file if proxy_settings.use_https else None
        ),
    )


if __name__ == "__main__":
    run()
