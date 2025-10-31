import json
import time
from typing import AsyncGenerator

import tiktoken
from aioitertools import enumerate as aio_enumerate
import os
from pathlib import Path
import time
import asyncio
import random
import base64
import struct
import statistics
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from gigachat.models import FunctionParameters, Function
from openai.pagination import AsyncPage
from openai.types import Model as OpenAIModel

from gpt2giga.utils import exceptions_handler
from gpt2giga.auth import TokenAwareClient
from openai.types.responses import ResponseTextDeltaEvent
import uuid

router = APIRouter()
def _load_embeddings_config(app) -> dict:
    cfg = getattr(app.state, "embeddings_config", None)
    if cfg is not None:
        return cfg
    # Resolve config path: env override or default to project config/embeddings.json
    env_path = os.getenv("GPT2GIGA_EMBEDDINGS_CONFIG_FILE", "").strip()
    if env_path:
        path = Path(env_path)
    else:
        path = (Path(__file__).resolve().parent.parent / "config" / "embeddings.json")
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = {}
    except Exception:
        raw = {}

    # Normalize array format -> map {MODEL: limit}
    cfg_map: dict[str, int] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("model")
                limit = item.get("limit")
                if name and isinstance(limit, int):
                    cfg_map[name.upper()] = int(limit)
    elif isinstance(raw, dict):
        src = raw.get("limits") if isinstance(raw.get("limits"), dict) else raw
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(v, int):
                    cfg_map[str(k).upper()] = int(v)

    app.state.embeddings_config = cfg_map
    return cfg_map


def _get_token_limit_for_model(app, model_name: str) -> int:
    cfg = _load_embeddings_config(app)
    if not isinstance(cfg, dict):
        return 0
    if not model_name:
        return int(cfg.get("DEFAULT", 0) or 0)
    key = model_name.upper()
    if key in cfg:
        return int(cfg[key])
    norm = "".join(ch if ch.isalnum() else "_" for ch in model_name).upper()
    return int(cfg.get(norm, 0) or 0)

def _get_rate_state(app):
    state = getattr(app.state, "_emb_rate", None)
    if state is None:
        state = {
            "delay_ms": int(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_MIN_MS", "0") or 0),
            "min_ms": int(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_MIN_MS", "0") or 0),
            "max_ms": int(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_MAX_MS", "1000") or 1000),
            "backoff": float(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_BACKOFF_FACTOR", "2") or 2),
            "success_window": int(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_SUCCESS_WINDOW", "10") or 10),
            "drop_after": int(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_DROP_TO_AVG_AFTER", "10") or 10),
            "success_count": 0,
            # Switch to median-based stability reference
            "stable_med_ms": 0.0,
            "samples": [],
            "max_samples": int(os.getenv("GPT2GIGA_EMBEDDINGS_RATE_MEDIAN_SAMPLES", "21") or 21),
        }
        app.state._emb_rate = state
    return state


def _get_metrics_state(app):
    state = getattr(app.state, "_metrics", None)
    if state is None:
        state = {
            "emb_total": 0,
            "emb_timeouts": 0,
            "emb_throttles": 0,
            "emb_retries": 0,
            "pool_rebuilds": 0,
            "durations_ms": [],
            "max_samples": 200,
            "emb_total_time_ms": 0.0,
            "queue_enqueued": 0,
            "queue_processed": 0,
            "queue_dropped": 0,
            "queue_size": 0,
        }
        app.state._metrics = state
    return state


async def _ensure_emb_queue(app):
    if getattr(app.state, "_embq_inited", False):
        return
    try:
        enabled = os.getenv("GPT2GIGA_EMB_QUEUE_ENABLED", "true").lower() == "true"
        if not enabled:
            app.state._embq_inited = True
            return
        size = int(os.getenv("GPT2GIGA_EMB_QUEUE_SIZE", "100") or 100)
        workers = int(os.getenv("GPT2GIGA_EMB_WORKERS", "2") or 2)
        rps = float(os.getenv("GPT2GIGA_EMB_RATE_RPS", "0") or 0)
        app.state._emb_queue = asyncio.Queue(maxsize=max(1, size))
        app.state._emb_rate_rps = max(0.0, rps)
        app.state._emb_next_time = 0.0

        async def worker_loop(idx: int):
            while True:
                try:
                    request, fut = await app.state._emb_queue.get()
                    try:
                        # simple token-bucket: sleep to enforce RPS
                        rps_local = float(getattr(app.state, "_emb_rate_rps", 0.0) or 0.0)
                        if rps_local > 0:
                            interval = 1.0 / rps_local
                            now = asyncio.get_event_loop().time()
                            next_t = max(now, getattr(app.state, "_emb_next_time", 0.0))
                            delay = max(0.0, next_t - now)
                            if delay > 0:
                                await asyncio.sleep(delay)
                            app.state._emb_next_time = (asyncio.get_event_loop().time()) + interval
                        result = await _embeddings_async(request)
                        if not fut.cancelled():
                            fut.set_result(result)
                        # Track queue metrics
                        try:
                            metrics = _get_metrics_state(app)
                            metrics["queue_processed"] = metrics.get("queue_processed", 0) + 1
                            metrics["queue_size"] = app.state._emb_queue.qsize()
                        except Exception:
                            pass
                    except Exception as e:
                        if not fut.cancelled():
                            fut.set_exception(e)
                        # Track queue metrics even on error
                        try:
                            metrics = _get_metrics_state(app)
                            metrics["queue_processed"] = metrics.get("queue_processed", 0) + 1
                            metrics["queue_size"] = app.state._emb_queue.qsize()
                        except Exception:
                            pass
                    finally:
                        app.state._emb_queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(0.1)

        app.state._emb_workers = [asyncio.create_task(worker_loop(i)) for i in range(workers)]
        app.state._embq_inited = True
    except Exception:
        app.state._embq_inited = True


async def _call_embeddings_with_retry(app, texts: list[str], model: str):
    # Use embeddings-specific client with separate connection pool
    client = getattr(app.state, "client_embeddings", app.state.client)
    """Call embeddings with adaptive retries on throttling (429/503).
    Also adapts pacing delay and stabilizes to last average when no rejects occur.
    Controls via env:
      - GPT2GIGA_EMBEDDINGS_MAX_RETRIES (default 3)
      - GPT2GIGA_EMBEDDINGS_BACKOFF_BASE_MS (default 200)
      - GPT2GIGA_EMBEDDINGS_BACKOFF_MAX_MS (default 5000)
      - GPT2GIGA_EMBEDDINGS_RATE_* knobs (see _get_rate_state)
    """
    max_retries = int(os.getenv("GPT2GIGA_EMBEDDINGS_MAX_RETRIES", "3") or 3)
    base_ms = int(os.getenv("GPT2GIGA_EMBEDDINGS_BACKOFF_BASE_MS", "200") or 200)
    max_ms = int(os.getenv("GPT2GIGA_EMBEDDINGS_BACKOFF_MAX_MS", "5000") or 5000)
    rate = _get_rate_state(app)
    adaptive_on = str(os.getenv("GPT2GIGA_EMBEDDINGS_ADAPTIVE", "true")).lower() == "true"

    # Timeout controls based on median * factor
    t_factor = float(os.getenv("GPT2GIGA_EMBEDDINGS_TIMEOUT_FACTOR", "2") or 2)
    t_min_ms = int(os.getenv("GPT2GIGA_EMBEDDINGS_TIMEOUT_MIN_MS", "1000") or 1000)
    t_max_ms = int(os.getenv("GPT2GIGA_EMBEDDINGS_TIMEOUT_MAX_MS", "60000") or 60000)
    timeout_max_retries = int(os.getenv("GPT2GIGA_EMBEDDINGS_TIMEOUT_MAX_RETRIES", "0") or 0)  # 0=infinite
    timeouts = 0
    metrics = _get_metrics_state(app)

    for attempt in range(max_retries + 1):
        # Respect current pacing delay
        if rate["delay_ms"] > 0:
            await asyncio.sleep(rate["delay_ms"] / 1000.0)
        try:
            t0 = time.monotonic()
            stable_med = float(rate.get("stable_med_ms", 0.0) or 0.0)
            timeout_ms = max(t_min_ms, min(t_max_ms, int(stable_med * t_factor) if stable_med > 0 else t_min_ms))
            # request-id span
            req_id = str(uuid.uuid4())
            logger = getattr(app.state, "logger", None)
            if logger:
                logger.debug(
                    "emb span start id=%s timeout_ms=%d texts=%d", req_id, timeout_ms, len(texts)
                )
            result = await asyncio.wait_for(client.aembeddings(texts=texts, model=model), timeout=timeout_ms / 1000.0)
            # Normalize response: convert Pydantic model to dict if needed
            if not isinstance(result, dict):
                try:
                    # Try Pydantic v1 style (.dict())
                    if hasattr(result, "dict"):
                        result = result.dict()
                    # Try Pydantic v2 style (.model_dump())
                    elif hasattr(result, "model_dump"):
                        result = result.model_dump()
                except Exception as e:
                    logger = getattr(app.state, "logger", None)
                    if logger:
                        logger.warning("Failed to convert embeddings response to dict: %s (type=%s, error=%s)", result, type(result).__name__, str(e))
            # Success: update metrics
            elapsed = (time.monotonic() - t0) * 1000.0
            try:
                metrics["emb_success"] = metrics.get("emb_success", 0) + 1
                # Track cumulative time for average calculation (always)
                metrics["emb_total_time_ms"] = metrics.get("emb_total_time_ms", 0.0) + elapsed
            except Exception:
                pass
            # Success: update median-based stability and success counter
            if adaptive_on:
                # metrics: track duration
                try:
                    durations = metrics.get("durations_ms", [])
                    durations.append(elapsed)
                    if len(durations) > metrics.get("max_samples", 200):
                        durations[:] = durations[-metrics.get("max_samples", 200):]
                    metrics["durations_ms"] = durations
                    metrics["emb_total"] = metrics.get("emb_total", 0) + 1
                except Exception:
                    pass
                samples = rate.get("samples", [])
                max_s = int(rate.get("max_samples", 21) or 21)
                samples.append(elapsed)
                if len(samples) > max_s:
                    samples = samples[-max_s:]
                sorted_s = sorted(samples)
                mid = len(sorted_s) // 2
                if len(sorted_s) % 2 == 1:
                    median = sorted_s[mid]
                else:
                    median = 0.5 * (sorted_s[mid - 1] + sorted_s[mid])
                rate["samples"] = samples
                rate["stable_med_ms"] = median
                rate["success_count"] = min(rate["success_window"], rate["success_count"] + 1)
                # After sustained success, set delay to 2 × median (halve frequency)
                if rate["success_count"] >= rate["drop_after"] and rate["delay_ms"] > rate.get("stable_med_ms", 0):
                    target = int((rate.get("stable_med_ms", 0) or 0) * 2)
                    rate["delay_ms"] = max(rate["min_ms"], target)
            if logger:
                total_ms = (time.monotonic() - t0) * 1000.0
                logger.debug("emb span end id=%s total_ms=%.3f upstream_ms=%.3f", req_id, total_ms, elapsed)
            # slow logs
            try:
                slow_total = int(os.getenv("SLOW_EMB_TOTAL_MS", "2000") or 2000)
                slow_up = int(os.getenv("SLOW_EMB_UPSTREAM_MS", "1500") or 1500)
                if total_ms >= slow_total or elapsed >= slow_up:
                    if logger:
                        logger.warning(
                            "slow emb id=%s total_ms=%.3f upstream_ms=%.3f texts=%d", req_id, total_ms, elapsed, len(texts)
                        )
            except Exception:
                pass
            return result
        except asyncio.TimeoutError:
            timeouts += 1
            try:
                logger = getattr(app.state, "logger", None)
                if logger:
                    logger.info(
                        "Embeddings request timed out after %d ms (timeouts=%d)",
                        timeout_ms,
                        timeouts,
                    )
                metrics["emb_timeouts"] = metrics.get("emb_timeouts", 0) + 1
            except Exception:
                pass
            # Optionally drop and rebuild client to force socket close
            try:
                if os.getenv("GPT2GIGA_CLOSE_SOCKET_ON_TIMEOUT", "false").lower() == "true":
                    # Close embeddings-specific http client in background and swap immediately
                    http_client = getattr(app.state, "http_client_embeddings", None)
                    if http_client:
                        try:
                            asyncio.create_task(http_client.aclose())
                        except Exception:
                            pass
                        # Rebuild embeddings-specific pooled client
                        build_http = getattr(app.state, "build_http_client", None)
                        if build_http:
                            app.state.http_client_embeddings = build_http("embeddings")
                            setattr(app.state, "_force_conn_close", True)
                    # Rebuild embeddings-specific GigaChat client wrapper
                    build = getattr(app.state, "build_client", None)
                    tman = getattr(app.state, "token_manager", None)
                    if build and tman:
                        new_gc = build("embeddings")
                        app.state.gigachat_client_embeddings = new_gc
                        app.state.client_embeddings = TokenAwareClient(tman, lambda: build("embeddings"), new_gc)
                    metrics["pool_rebuilds"] = metrics.get("pool_rebuilds", 0) + 1
            except Exception:
                pass
            if timeout_max_retries == 0 or timeouts <= timeout_max_retries:
                continue
            raise
        except Exception as e:
            # Try to detect throttling/status
            status = None
            try:
                import gigachat  # type: ignore
                if isinstance(e, gigachat.exceptions.ResponseError) and len(e.args) == 4:
                    _, status_code, _, _ = e.args
                    status = int(status_code)
            except Exception:
                pass

            if status in (429, 503):
                # Increase delay on throttling (bounded). Reset successes.
                rate["success_count"] = 0
                boosted = int(max(rate["min_ms"], max(rate["delay_ms"], rate["min_ms"]) * rate["backoff"]))
                rate["delay_ms"] = min(rate["max_ms"], boosted)
                if attempt < max_retries:
                    # exponential backoff with jitter for immediate retry
                    delay_ms = min(max_ms, base_ms * (2 ** attempt))
                    jitter_ms = random.randint(0, delay_ms // 2)
                    await asyncio.sleep((delay_ms + jitter_ms) / 1000.0)
                    try:
                        metrics["emb_throttles"] = metrics.get("emb_throttles", 0) + 1
                        metrics["emb_retries"] = metrics.get("emb_retries", 0) + 1
                    except Exception:
                        pass
                    continue
            # Non-throttle or retries exhausted
            raise

    # Fallback - use embeddings-specific client
    client = getattr(app.state, "client_embeddings", app.state.client)
    result = await client.aembeddings(texts=texts, model=model)
    # Normalize response: convert Pydantic model to dict if needed
    if not isinstance(result, dict):
        try:
            # Try Pydantic v1 style (.dict())
            if hasattr(result, "dict"):
                result = result.dict()
            # Try Pydantic v2 style (.model_dump())
            elif hasattr(result, "model_dump"):
                result = result.model_dump()
        except Exception as e:
            logger = getattr(app.state, "logger", None)
            if logger:
                logger.warning("Failed to convert embeddings response to dict: %s (type=%s, error=%s)", result, type(result).__name__, str(e))
    return result

def _aggregate_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    sums = [0.0] * length
    for vec in vectors:
        # handle mismatched lengths conservatively
        l = min(length, len(vec))
        for i in range(l):
            sums[i] += float(vec[i])
    count = float(len(vectors))
    return [sums[i] / count for i in range(length)]


def _encode_embedding_base64(vec: list[float], urlsafe: bool = False) -> str:
    if not vec:
        return ""
    try:
        packed = struct.pack("<%sf" % len(vec), *[float(x) for x in vec])
    except Exception:
        # fallback: try best-effort cast
        casted = []
        for x in vec:
            try:
                casted.append(float(x))
            except Exception:
                casted.append(0.0)
        packed = struct.pack("<%sf" % len(casted), *casted)
    if urlsafe:
        return base64.urlsafe_b64encode(packed).decode("ascii")
    return base64.b64encode(packed).decode("ascii")


@router.get("/health", response_class=Response)
@exceptions_handler
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@router.get("/ping", response_class=Response)
@router.post("/ping", response_class=Response)
@exceptions_handler
async def ping() -> Response:
    return await health()


@router.get("/models")
@exceptions_handler
async def show_available_models(raw_request: Request):
    response = await raw_request.app.state.client.aget_models()
    models = [i.dict(by_alias=True) for i in response.data]
    current_timestamp = int(time.time())
    for model in models:
        model["created"] = current_timestamp
    models = [OpenAIModel(**model) for model in models]
    model_page = AsyncPage(data=models, object=response.object_)
    return model_page


@router.get("/models/{model}")
@exceptions_handler
async def get_model(model: str, request: Request):
    response = await request.app.state.client.aget_model(model=model)
    model = response.dict(by_alias=True)
    model["created"] = int(time.time())
    return OpenAIModel(**model)


@router.post("/chat/completions")
@exceptions_handler
async def chat_completions(request: Request):
    try:
        request.app.state.last_activity = asyncio.get_event_loop().time()
        # Reset pool restart tracking on new activity
        if hasattr(request.app.state, "_pool_restart_idle_ms"):
            request.app.state._pool_restart_idle_ms = 0
    except Exception:
        pass
    data = await request.json()
    # Debug-log POST body
    try:
        logger = getattr(request.app.state, "logger", None)
        if logger and logger.isEnabledFor(10):  # DEBUG
            logger.debug("POST /chat/completions body: %s", json.dumps(data))
    except Exception:
        pass
    stream = data.get("stream", False)
    is_tool_call = "tools" in data
    is_response_api = "input" in data
    if is_tool_call:
        data["functions"] = []
        for tool in data.get("tools", []):
            if tool.get("function"):
                function = tool["function"]
                giga_function = Function(
                    name=function["name"],
                    description=function["description"],
                    parameters=FunctionParameters(**function["parameters"]),
                )
            else:
                giga_function = Function(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=FunctionParameters(**tool["parameters"]),
                )
            data["functions"].append(giga_function)
    chat_messages = request.app.state.request_transformer.send_to_gigachat(data)
    # Use chat-specific client with separate connection pool
    client_chat = getattr(request.app.state, "client_chat", request.app.state.client)
    if not stream:
        response = await client_chat.achat(chat_messages)
        if is_response_api:
            processed = request.app.state.response_processor.process_response_api(
                data, response, chat_messages.model, is_tool_call
            )
        else:
            processed = request.app.state.response_processor.process_response(
                response, chat_messages.model, is_tool_call
            )
        return processed
    else:

        async def stream_generator(is_response_api: bool) -> AsyncGenerator[str, None]:
            """
            Yields formatted SSE (Server-Sent Events) chunks
            as they arrive from the model.
            """
            # Coalescing settings (env-configurable):
            # - GPT2GIGA_STREAM_COALESCE_BYTES: integer, when buffered text >= bytes → flush
            #   (0 or unset disables byte-based coalescing)
            # - GPT2GIGA_STREAM_COALESCE_MS: integer milliseconds, maximum idle time before flush
            #   (0 or unset disables time-based coalescing)
            # If both are 0/unset, streaming behavior is passthrough (no coalescing).
            max_bytes = int(os.getenv("GPT2GIGA_STREAM_COALESCE_BYTES", "0") or 0)
            max_interval_ms = int(
                os.getenv("GPT2GIGA_STREAM_COALESCE_MS", "0") or 0
            )

            if is_response_api:
                buf = []
                last_flush = time.monotonic()
                last_seq = 0
                async for i, chunk in aio_enumerate(
                    client_chat.astream(chat_messages)
                ):
                    processed = request.app.state.response_processor.process_stream_chunk_response(
                        chunk, sequence_number=i
                    )

                    def flush_buffer():
                        nonlocal buf, last_seq, last_flush
                        if not buf:
                            return None
                        merged = "".join(buf)
                        buf.clear()
                        last_flush = time.monotonic()
                        event = ResponseTextDeltaEvent(
                            content_index=0,
                            delta=merged,
                            item_id=f"msg_{int(time.time()*1000)}",
                            output_index=0,
                            logprobs=[],
                            type="response.output_text.delta",
                            sequence_number=last_seq,
                        ).dict()
                        return event

                    if isinstance(processed, dict) and processed.get(
                        "type"
                    ) == "response.output_text.delta":
                        # Buffer textual delta
                        delta_text = processed.get("delta", "")
                        if max_bytes > 0 or max_interval_ms > 0:
                            if delta_text:
                                buf.append(delta_text)
                                last_seq = processed.get("sequence_number", i)
                            need_flush = False
                            if max_bytes > 0 and sum(len(x) for x in buf) >= max_bytes:
                                need_flush = True
                            if (
                                not need_flush
                                and max_interval_ms > 0
                                and (time.monotonic() - last_flush)
                                * 1000.0
                                >= max_interval_ms
                            ):
                                need_flush = True
                            if need_flush:
                                merged_event = flush_buffer()
                                if merged_event:
                                    yield f"data: {json.dumps(merged_event)}\n\n"
                        else:
                            yield f"data: {json.dumps(processed)}\n\n"
                    else:
                        # Non-delta event: flush buffer first
                        merged_event = flush_buffer()
                        if merged_event:
                            yield f"data: {json.dumps(merged_event)}\n\n"
                        yield f"data: {json.dumps(processed)}\n\n"

                # Final flush
                if max_bytes > 0 or max_interval_ms > 0:
                    if buf:
                        merged_event = ResponseTextDeltaEvent(
                            content_index=0,
                            delta="".join(buf),
                            item_id=f"msg_{int(time.time()*1000)}",
                            output_index=0,
                            logprobs=[],
                            type="response.output_text.delta",
                            sequence_number=last_seq,
                        ).dict()
                        yield f"data: {json.dumps(merged_event)}\n\n"
            else:
                buf = []
                last_flush = time.monotonic()
                # Generate consistent stream identifiers per OpenAI spec
                stream_id = f"chatcmpl-{uuid.uuid4()}"
                stream_fingerprint = f"fp_{uuid.uuid4()}"
                stream_created = int(time.time())
                
                async for chunk in client_chat.astream(
                    chat_messages
                ):
                    processed = (
                        request.app.state.response_processor.process_stream_chunk(
                            chunk,
                            chat_messages.model,
                            is_tool_call="tools" in chat_messages,
                            stream_id=stream_id,
                            stream_fingerprint=stream_fingerprint,
                            stream_created=stream_created,
                        )
                    )
                    content = (
                        processed.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )

                    def flush_buffer_nonresp():
                        nonlocal buf, last_flush
                        if not buf:
                            return None
                        merged = "".join(buf)
                        buf.clear()
                        last_flush = time.monotonic()
                        # clone processed shape with merged content
                        out = processed.copy()
                        out["choices"] = [
                            {**processed["choices"][0], "delta": {"content": merged}}
                        ]
                        return out

                    # If no coalescing requested -> pass through
                    if max_bytes == 0 and max_interval_ms == 0:
                        # Remove usage from non-final chunks per OpenAI spec
                        chunk_to_yield = processed.copy()
                        is_final_passthrough = any(
                            choice.get("finish_reason") is not None 
                            for choice in processed.get("choices", [])
                        ) or "usage" in processed
                        if not is_final_passthrough and "usage" in chunk_to_yield:
                            del chunk_to_yield["usage"]
                        yield f"data: {json.dumps(chunk_to_yield)}\n\n"
                        continue

                    # Check if this is the final chunk (has finish_reason or usage)
                    is_final = any(
                        choice.get("finish_reason") is not None 
                        for choice in processed.get("choices", [])
                    ) or "usage" in processed
                    
                    # If this is the final chunk, flush buffer and emit it
                    if is_final:
                        merged_out = flush_buffer_nonresp()
                        if merged_out:
                            yield f"data: {json.dumps(merged_out)}\n\n"
                        yield f"data: {json.dumps(processed)}\n\n"
                        continue
                    
                    # If this chunk carries tool_calls etc., flush buffer and emit as-is
                    delta_obj = processed.get("choices", [{}])[0].get("delta", {})
                    if any(k in delta_obj for k in ("tool_calls", "function_call")):
                        merged_out = flush_buffer_nonresp()
                        if merged_out:
                            yield f"data: {json.dumps(merged_out)}\n\n"
                        yield f"data: {json.dumps(processed)}\n\n"
                        continue

                    if content:
                        buf.append(content)
                    need_flush = False
                    if max_bytes > 0 and sum(len(x) for x in buf) >= max_bytes:
                        need_flush = True
                    if (
                        not need_flush
                        and max_interval_ms > 0
                        and (time.monotonic() - last_flush) * 1000.0 >= max_interval_ms
                    ):
                        need_flush = True
                    if need_flush:
                        merged_out = flush_buffer_nonresp()
                        if merged_out:
                            yield f"data: {json.dumps(merged_out)}\n\n"
                    # Yield chunk without usage (will be in final chunk per OpenAI spec)
                    # Remove usage from non-final chunks
                    chunk_to_yield = processed.copy()
                    if "usage" in chunk_to_yield:
                        del chunk_to_yield["usage"]
                    yield f"data: {json.dumps(chunk_to_yield)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(is_response_api), media_type="text/event-stream"
        )


 


async def _embeddings_async(request: Request):
    data = await request.json()
    # Debug-log POST body
    try:
        logger = getattr(request.app.state, "logger", None)
        if logger and logger.isEnabledFor(10):  # DEBUG
            logger.debug("POST /embeddings body: %s", json.dumps(data))
    except Exception:
        pass
    inputs = data.get("input", [])
    emb_format = (data.get("encoding_format") or "").lower().strip()
    use_b64 = emb_format in ("base64", "base64url")
    urlsafe = emb_format == "base64url"
    gpt_model = data.get("model", None)

    # Validate inputs
    if not inputs:
        return Response(
            status_code=400,
            content=json.dumps({"error": {"message": "input is required and cannot be empty", "type": "invalid_request_error"}}),
            media_type="application/json",
        )
    
    # Validate model
    if not gpt_model:
        return Response(
            status_code=400,
            content=json.dumps({"error": {"message": "model is required", "type": "invalid_request_error"}}),
            media_type="application/json",
        )

    # Process inputs
    if isinstance(inputs, list):
        if not inputs:  # Empty list
            return Response(
                status_code=400,
                content=json.dumps({"error": {"message": "input array cannot be empty", "type": "invalid_request_error"}}),
                media_type="application/json",
            )
        new_inputs = []
        try:
            # Check if first element is int (token IDs) or list[int]
            if isinstance(inputs[0], int):
                # Single token ID sequence - decode to single string
                new_inputs = [tiktoken.encoding_for_model(gpt_model).decode(inputs)]
            elif isinstance(inputs[0], list) and inputs[0] and isinstance(inputs[0][0], int):
                # List of token ID sequences - decode each
                enc = tiktoken.encoding_for_model(gpt_model)
                for row in inputs:
                    if isinstance(row, list) and all(isinstance(x, int) for x in row):
                        new_inputs.append(enc.decode(row))
                    else:
                        return Response(
                            status_code=400,
                            content=json.dumps({"error": {"message": "invalid input format: expected string array or token ID arrays", "type": "invalid_request_error"}}),
                            media_type="application/json",
                        )
            else:
                # List of strings - pass through
                for row in inputs:
                    if not isinstance(row, str):
                        return Response(
                            status_code=400,
                            content=json.dumps({"error": {"message": "invalid input format: expected string array", "type": "invalid_request_error"}}),
                            media_type="application/json",
                        )
                    # Allow empty strings - backend will handle them
                    new_inputs.append(row)
        except Exception as e:
            logger = getattr(request.app.state, "logger", None)
            if logger:
                logger.warning("Error processing input format: %s", str(e))
            return Response(
                status_code=400,
                content=json.dumps({"error": {"message": f"invalid input format: {str(e)}", "type": "invalid_request_error"}}),
                media_type="application/json",
            )
    else:
        # Single input - convert to list
        if not isinstance(inputs, str):
            return Response(
                status_code=400,
                content=json.dumps({"error": {"message": "input must be a string or array of strings", "type": "invalid_request_error"}}),
                media_type="application/json",
            )
        new_inputs = [inputs]

    # Apply per-model limits: chunk inputs to max tokens per model, then aggregate
    try:
        limit = _get_token_limit_for_model(request.app, gpt_model)
    except Exception as e:
        logger = getattr(request.app.state, "logger", None)
        if logger:
            logger.warning("Error getting token limit for model %s: %s", gpt_model, str(e))
        limit = 0  # Fallback: no limit
    
    if not limit or limit <= 0:
        resp = await _call_embeddings_with_retry(
            request.app, new_inputs, request.app.state.config.proxy_settings.embeddings
        )
        # Validate response
        if not isinstance(resp, dict):
            logger = getattr(request.app.state, "logger", None)
            if logger:
                logger.error("Invalid response format from embeddings backend: %s", type(resp))
            return Response(
                status_code=500,
                content=json.dumps({"error": {"message": "invalid response from embeddings service", "type": "internal_error"}}),
                media_type="application/json",
            )
        resp_data = resp.get("data")
        if not isinstance(resp_data, list) or len(resp_data) != len(new_inputs):
            logger = getattr(request.app.state, "logger", None)
            if logger:
                logger.error("Response data mismatch: expected %d items, got %d", len(new_inputs), len(resp_data) if isinstance(resp_data, list) else 0)
            return Response(
                status_code=500,
                content=json.dumps({"error": {"message": f"response data mismatch: expected {len(new_inputs)} items", "type": "internal_error"}}),
                media_type="application/json",
            )
        if use_b64:
            out = {"data": [], "model": resp.get("model", gpt_model)}
            for item in resp_data:
                vec = item.get("embedding", [])
                if not vec or not isinstance(vec, list):
                    logger = getattr(request.app.state, "logger", None)
                    if logger:
                        logger.error("Invalid embedding vector in response: %s", type(vec))
                    return Response(
                        status_code=500,
                        content=json.dumps({"error": {"message": "invalid embedding vector in response", "type": "internal_error"}}),
                        media_type="application/json",
                    )
                b64 = _encode_embedding_base64(vec, urlsafe=urlsafe)
                # Match OpenAI format: base64 string in "embedding" field, not "embedding_b64"
                out["data"].append(
                    {
                        "embedding": b64,
                        "index": item.get("index", len(out["data"])),
                    }
                )
            return out
        return resp

    try:
        enc = tiktoken.encoding_for_model(gpt_model)
    except Exception as e:
        logger = getattr(request.app.state, "logger", None)
        if logger:
            logger.warning("Error getting encoding for model %s: %s", gpt_model, str(e))
        return Response(
            status_code=400,
            content=json.dumps({"error": {"message": f"invalid model or encoding error: {str(e)}", "type": "invalid_request_error"}}),
            media_type="application/json",
        )
    
    per_input_chunks: list[list[str]] = []
    flat_chunks: list[str] = []
    try:
        for text in new_inputs:
            ids = enc.encode(text)
            if len(ids) <= limit:
                chunks = [text]
            else:
                chunks = []
                for i in range(0, len(ids), limit):
                    chunk_ids = ids[i : i + limit]
                    chunks.append(enc.decode(chunk_ids))
            per_input_chunks.append(chunks)
            flat_chunks.extend(chunks)
    except Exception as e:
        logger = getattr(request.app.state, "logger", None)
        if logger:
            logger.warning("Error encoding/chunking inputs: %s", str(e))
        return Response(
            status_code=400,
            content=json.dumps({"error": {"message": f"error processing inputs: {str(e)}", "type": "invalid_request_error"}}),
            media_type="application/json",
        )

    chunk_resp = await _call_embeddings_with_retry(
        request.app, flat_chunks, request.app.state.config.proxy_settings.embeddings
    )
    # Validate chunk response
    if not isinstance(chunk_resp, dict) or not isinstance(chunk_resp.get("data"), list):
        logger = getattr(request.app.state, "logger", None)
        if logger:
            logger.error("Invalid chunk response format from embeddings backend")
        return Response(
            status_code=500,
            content=json.dumps({"error": {"message": "invalid response from embeddings service", "type": "internal_error"}}),
            media_type="application/json",
        )
    vectors = [row.get("embedding", []) for row in chunk_resp.get("data", [])]
    if len(vectors) != len(flat_chunks):
        logger = getattr(request.app.state, "logger", None)
        if logger:
            logger.error("Chunk response mismatch: expected %d vectors, got %d", len(flat_chunks), len(vectors))
        return Response(
            status_code=500,
            content=json.dumps({"error": {"message": f"response data mismatch: expected {len(flat_chunks)} vectors, got {len(vectors)}", "type": "internal_error"}}),
            media_type="application/json",
        )

    result_data = []
    cursor = 0
    for idx, chunks in enumerate(per_input_chunks):
        n = len(chunks)
        sub = vectors[cursor : cursor + n]
        cursor += n
        merged = _aggregate_vectors(sub)
        if not merged or len(merged) == 0:
            logger = getattr(request.app.state, "logger", None)
            if logger:
                logger.error("Empty merged vector for input %d (chunks=%d)", idx, n)
            return Response(
                status_code=500,
                content=json.dumps({"error": {"message": f"failed to generate embedding for input {idx}", "type": "internal_error"}}),
                media_type="application/json",
            )
        if use_b64:
            b64 = _encode_embedding_base64(merged, urlsafe=urlsafe)
            # Match OpenAI format: base64 string in "embedding" field, not "embedding_b64"
            result_data.append(
                {
                    "embedding": b64,
                    "index": idx,
                }
            )
        else:
            result_data.append({"embedding": merged, "index": idx})

    return {"data": result_data, "model": chunk_resp.get("model")}


@router.get("/metrics")
async def metrics(request: Request):
    """Expose simple text metrics."""
    m = _get_metrics_state(request.app)
    lines = []
    total = int(m.get("emb_total", 0) or 0)
    timeouts = int(m.get("emb_timeouts", 0) or 0)
    throttles = int(m.get("emb_throttles", 0) or 0)
    retries = int(m.get("emb_retries", 0) or 0)
    rebuilds = int(m.get("pool_rebuilds", 0) or 0)
    durs = list(m.get("durations_ms", []) or [])
    p50 = p95 = p99 = 0.0
    if durs:
        try:
            d_sorted = sorted(durs)
            p50 = statistics.median(d_sorted)
            idx95 = max(0, int(0.95 * (len(d_sorted) - 1)))
            idx99 = max(0, int(0.99 * (len(d_sorted) - 1)))
            p95 = float(d_sorted[idx95])
            p99 = float(d_sorted[idx99])
        except Exception:
            pass
    lines.append(f"emb_total {total}")
    lines.append(f"emb_timeouts {timeouts}")
    lines.append(f"emb_success {int(m.get('emb_success', 0) or 0)}")
    lines.append(f"emb_throttles {throttles}")
    lines.append(f"emb_retries {retries}")
    lines.append(f"pool_rebuilds {rebuilds}")
    lines.append(f"emb_p50_ms {p50:.3f}")
    lines.append(f"emb_p95_ms {p95:.3f}")
    lines.append(f"emb_p99_ms {p99:.3f}")
    # Average time per embedding since start
    try:
        total_time_ms = float(m.get("emb_total_time_ms", 0.0) or 0.0)
        success_count = int(m.get("emb_success", 0) or 0)
        if success_count > 0:
            avg_ms = total_time_ms / float(success_count)
            lines.append(f"emb_avg_ms {avg_ms:.3f}")
        else:
            lines.append(f"emb_avg_ms 0.000")
    except Exception:
        lines.append(f"emb_avg_ms 0.000")
    # Queue metrics
    lines.append(f"queue_enqueued {int(m.get('queue_enqueued', 0) or 0)}")
    lines.append(f"queue_processed {int(m.get('queue_processed', 0) or 0)}")
    lines.append(f"queue_dropped {int(m.get('queue_dropped', 0) or 0)}")
    # Current queue size (live)
    try:
        if hasattr(request.app.state, "_emb_queue"):
            lines.append(f"queue_size {request.app.state._emb_queue.qsize()}")
        else:
            lines.append(f"queue_size {int(m.get('queue_size', 0) or 0)}")
    except Exception:
        lines.append(f"queue_size {int(m.get('queue_size', 0) or 0)}")
    return Response("\n".join(lines) + "\n", media_type="text/plain")


@router.post("/embeddings")
@exceptions_handler
async def embeddings(request: Request):
    try:
        request.app.state.last_activity = asyncio.get_event_loop().time()
        # Reset pool restart tracking on new activity
        if hasattr(request.app.state, "_pool_restart_idle_ms"):
            request.app.state._pool_restart_idle_ms = 0
    except Exception:
        pass
    # If queueing enabled, enqueue and await result
    await _ensure_emb_queue(request.app)
    if hasattr(request.app.state, "_emb_queue"):
        q: asyncio.Queue = request.app.state._emb_queue
        metrics = _get_metrics_state(request.app)
        if q.full():
            try:
                metrics["queue_dropped"] = metrics.get("queue_dropped", 0) + 1
            except Exception:
                pass
            return Response(status_code=429, content=json.dumps({"error": "queue_full"}), media_type="application/json")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await q.put((request, fut))
        try:
            metrics["queue_enqueued"] = metrics.get("queue_enqueued", 0) + 1
            metrics["queue_size"] = q.qsize()
        except Exception:
            pass
        return await fut
    # Fallback direct processing
    return await _embeddings_async(request)


 


 


@router.post("/responses")
@exceptions_handler
async def responses(request: Request):
    return await chat_completions(request)
