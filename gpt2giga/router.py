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
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from gigachat.models import FunctionParameters, Function
from openai.pagination import AsyncPage
from openai.types import Model as OpenAIModel

from gpt2giga.utils import exceptions_handler
from gpt2giga.auth import TokenAwareClient
from openai.types.responses import ResponseTextDeltaEvent

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


async def _call_embeddings_with_retry(app, texts: list[str], model: str):
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

    for attempt in range(max_retries + 1):
        # Respect current pacing delay
        if rate["delay_ms"] > 0:
            await asyncio.sleep(rate["delay_ms"] / 1000.0)
        try:
            t0 = time.monotonic()
            stable_med = float(rate.get("stable_med_ms", 0.0) or 0.0)
            timeout_ms = max(t_min_ms, min(t_max_ms, int(stable_med * t_factor) if stable_med > 0 else t_min_ms))
            result = await asyncio.wait_for(
                app.state.client.aembeddings(texts=texts, model=model),
                timeout=timeout_ms / 1000.0,
            )
            # Success: update median-based stability and success counter
            if adaptive_on:
                elapsed = (time.monotonic() - t0) * 1000.0
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
                # After sustained success, decrease delay to last stable median (bounded)
                if rate["success_count"] >= rate["drop_after"] and rate["delay_ms"] > rate.get("stable_med_ms", 0):
                    rate["delay_ms"] = max(rate["min_ms"], int(rate.get("stable_med_ms", 0)))
            return result
        except asyncio.TimeoutError:
            timeouts += 1
            # Optionally drop and rebuild client to force socket close
            try:
                if os.getenv("GPT2GIGA_CLOSE_SOCKET_ON_TIMEOUT", "false").lower() == "true":
                    build = getattr(app.state, "build_client", None)
                    tman = getattr(app.state, "token_manager", None)
                    if build and tman:
                        new_gc = build()
                        app.state.gigachat_client = new_gc
                        app.state.client = TokenAwareClient(tman, build, new_gc)
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
                    continue
            # Non-throttle or retries exhausted
            raise

    # Fallback
    return await app.state.client.aembeddings(texts=texts, model=model)

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
    if not stream:
        response = await request.app.state.client.achat(chat_messages)
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
                    request.app.state.client.astream(chat_messages)
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
                async for chunk in request.app.state.client.astream(
                    chat_messages
                ):
                    processed = (
                        request.app.state.response_processor.process_stream_chunk(
                            chunk,
                            chat_messages.model,
                            is_tool_call="tools" in chat_messages,
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

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(is_response_api), media_type="text/event-stream"
        )


 


@router.post("/embeddings")
@exceptions_handler
async def embeddings(request: Request):
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

    if isinstance(inputs, list):
        new_inputs = []
        if isinstance(inputs[0], int):  # List[int]:
            new_inputs = tiktoken.encoding_for_model(gpt_model).decode(inputs)
        else:
            for row in inputs:
                if isinstance(row, list):  # List[List[int]]
                    new_inputs.append(
                        tiktoken.encoding_for_model(gpt_model).decode(row)
                    )
                else:
                    new_inputs.append(row)
    else:
        new_inputs = [inputs]

    # Apply per-model limits: chunk inputs to max tokens per model, then aggregate
    limit = _get_token_limit_for_model(request.app, gpt_model)
    if not limit or limit <= 0:
        resp = await _call_embeddings_with_retry(
            request.app, new_inputs, request.app.state.config.proxy_settings.embeddings
        )
        if use_b64 and isinstance(resp, dict) and isinstance(resp.get("data"), list):
            out = {"data": [], "model": resp.get("model")}
            for item in resp["data"]:
                vec = item.get("embedding", [])
                b64 = _encode_embedding_base64(vec, urlsafe=urlsafe)
                out["data"].append(
                    {
                        "embedding_b64": b64,
                        "embedding_dtype": "float32",
                        "embedding_len": len(vec),
                        "index": item.get("index", 0),
                        "encoding": "base64url" if urlsafe else "base64",
                    }
                )
            return out
        return resp

    enc = tiktoken.encoding_for_model(gpt_model)
    per_input_chunks: list[list[str]] = []
    flat_chunks: list[str] = []
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

    chunk_resp = await _call_embeddings_with_retry(
        request.app, flat_chunks, request.app.state.config.proxy_settings.embeddings
    )
    vectors = [row.get("embedding", []) for row in chunk_resp.get("data", [])]

    result_data = []
    cursor = 0
    for idx, chunks in enumerate(per_input_chunks):
        n = len(chunks)
        sub = vectors[cursor : cursor + n]
        cursor += n
        merged = _aggregate_vectors(sub)
        if use_b64:
            b64 = _encode_embedding_base64(merged, urlsafe=urlsafe)
            result_data.append(
                {
                    "embedding_b64": b64,
                    "embedding_dtype": "float32",
                    "embedding_len": len(merged),
                    "index": idx,
                    "encoding": "base64url" if urlsafe else "base64",
                }
            )
        else:
            result_data.append({"embedding": merged, "index": idx})

    return {"data": result_data, "model": chunk_resp.get("model")}


 


 


@router.post("/responses")
@exceptions_handler
async def responses(request: Request):
    return await chat_completions(request)
