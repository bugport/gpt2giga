import json
import time
from typing import AsyncGenerator

import tiktoken
from aioitertools import enumerate as aio_enumerate
import os
import time
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from gigachat.models import FunctionParameters, Function
from openai.pagination import AsyncPage
from openai.types import Model as OpenAIModel

from gpt2giga.utils import exceptions_handler
from openai.types.responses import ResponseTextDeltaEvent

router = APIRouter()


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
    response = await raw_request.app.state.gigachat_client.aget_models()
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
    response = await request.app.state.gigachat_client.aget_model(model=model)
    model = response.dict(by_alias=True)
    model["created"] = int(time.time())
    return OpenAIModel(**model)


@router.post("/chat/completions")
@exceptions_handler
async def chat_completions(request: Request):
    data = await request.json()
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
        response = await request.app.state.gigachat_client.achat(chat_messages)
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
                    request.app.state.gigachat_client.astream(chat_messages)
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
                async for chunk in request.app.state.gigachat_client.astream(
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
    inputs = data.get("input", [])
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

    embeddings = await request.app.state.gigachat_client.aembeddings(
        texts=new_inputs, model=request.app.state.config.proxy_settings.embeddings
    )

    return embeddings


 


@router.post("/responses")
@exceptions_handler
async def responses(request: Request):
    return await chat_completions(request)
