from contextlib import asynccontextmanager
import json
from pathlib import Path
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

    def build_client():
        # Refresh token on each client build to ensure validity
        if token_manager.is_configured():
            token = token_manager.get_token()
            if token:
                config.gigachat_settings.access_token = token
        return GigaChat(**config.gigachat_settings.dict())

    # Expose client factory and token manager for runtime rebuilds
    app.state.build_client = build_client
    app.state.token_manager = token_manager

    app.state.gigachat_client = build_client()
    app.state.client = TokenAwareClient(token_manager, build_client, app.state.gigachat_client)

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
