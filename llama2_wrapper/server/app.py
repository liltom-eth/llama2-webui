import json
import multiprocessing
from re import compile, Match, Pattern
from threading import Lock
from functools import partial
from typing import Callable, Coroutine, Iterator, List, Optional, Tuple, Union, Dict
from typing_extensions import TypedDict, Literal

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import Depends, FastAPI, APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

from llama2_wrapper.model import LLAMA2_WRAPPER
from llama2_wrapper.types import (
    Completion,
    CompletionChunk,
    ChatCompletion,
    ChatCompletionChunk,
)


class Settings(BaseSettings):
    model_path: str = Field(
        default="",
        description="The path to the model to use for generating completions.",
    )
    backend_type: str = Field(
        default="llama.cpp",
        description="Backend for llama2, options: llama.cpp, gptq, transformers",
    )
    max_tokens: int = Field(default=4000, ge=1, description="Maximum context size.")
    load_in_8bit: bool = Field(
        default=False,
        description="`Whether to use bitsandbytes to run model in 8 bit mode (only for transformers models).",
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output to stderr.",
    )
    host: str = Field(default="localhost", description="API address")
    port: int = Field(default=8000, description="API port")
    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )


class ErrorResponse(TypedDict):
    """OpenAI style error response"""

    message: str
    type: str
    param: Optional[str]
    code: Optional[str]


class ErrorResponseFormatters:
    """Collection of formatters for error responses.

    Args:
        request (Union[CreateCompletionRequest, CreateChatCompletionRequest]):
            Request body
        match (Match[str]): Match object from regex pattern

    Returns:
        Tuple[int, ErrorResponse]: Status code and error response
    """

    @staticmethod
    def context_length_exceeded(
        request: Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for context length exceeded error"""

        context_window = int(match.group(2))
        prompt_tokens = int(match.group(1))
        completion_tokens = request.max_new_tokens
        if hasattr(request, "messages"):
            # Chat completion
            message = (
                "This model's maximum context length is {} tokens. "
                "However, you requested {} tokens "
                "({} in the messages, {} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        else:
            # Text completion
            message = (
                "This model's maximum context length is {} tokens, "
                "however you requested {} tokens "
                "({} in your prompt; {} for the completion). "
                "Please reduce your prompt; or completion length."
            )
        return 400, ErrorResponse(
            message=message.format(
                context_window,
                completion_tokens + prompt_tokens,
                prompt_tokens,
                completion_tokens,
            ),
            type="invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )

    @staticmethod
    def model_not_found(
        request: Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for model_not_found error"""

        model_path = str(match.group(1))
        message = f"The model `{model_path}` does not exist"
        return 400, ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=None,
            code="model_not_found",
        )


class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    # key: regex pattern for original error message from llama_cpp
    # value: formatter function
    pattern_and_formatters: Dict[
        "Pattern",
        Callable[
            [
                Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
                "Match[str]",
            ],
            Tuple[int, ErrorResponse],
        ],
    ] = {
        compile(
            r"Requested tokens \((\d+)\) exceed context window of (\d+)"
        ): ErrorResponseFormatters.context_length_exceeded,
        compile(
            r"Model path does not exist: (.+)"
        ): ErrorResponseFormatters.model_not_found,
    }

    def error_message_wrapper(
        self,
        error: Exception,
        body: Optional[
            Union[
                "CreateChatCompletionRequest",
                "CreateCompletionRequest",
            ]
        ] = None,
    ) -> Tuple[int, ErrorResponse]:
        """Wraps error message in OpenAI style error response"""

        if body is not None and isinstance(
            body,
            (
                CreateCompletionRequest,
                CreateChatCompletionRequest,
            ),
        ):
            # When text completion or chat completion
            for pattern, callback in self.pattern_and_formatters.items():
                match = pattern.search(str(error))
                if match is not None:
                    return callback(body, match)

        # Wrap other errors as internal server error
        return 500, ErrorResponse(
            message=str(error),
            type="internal_server_error",
            param=None,
            code=None,
        )

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[None, None, Response]]:
        """Defines custom route handler that catches exceptions and formats
        in OpenAI style error response"""

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except Exception as exc:
                json_body = await request.json()
                try:
                    if "messages" in json_body:
                        # Chat completion
                        body: Optional[
                            Union[
                                CreateChatCompletionRequest,
                                CreateCompletionRequest,
                            ]
                        ] = CreateChatCompletionRequest(**json_body)
                    elif "prompt" in json_body:
                        # Text completion
                        body = CreateCompletionRequest(**json_body)
                    # else:
                    #     # Embedding
                    #     body = CreateEmbeddingRequest(**json_body)
                except Exception:
                    # Invalid request body
                    body = None

                # Get proper error message from the exception
                (
                    status_code,
                    error_message,
                ) = self.error_message_wrapper(error=exc, body=body)
                return JSONResponse(
                    {"error": error_message},
                    status_code=status_code,
                )

        return custom_route_handler


router = APIRouter(route_class=RouteErrorHandler)

settings: Optional[Settings] = None
llama2: Optional[LLAMA2_WRAPPER] = None


def create_app(settings: Optional[Settings] = None):
    if settings is None:
        settings = Settings()
    app = FastAPI(
        title="llama2-wrapper Fast API",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    global llama2
    llama2 = LLAMA2_WRAPPER(
        model_path=settings.model_path,
        backend_type=settings.backend_type,
        max_tokens=settings.max_tokens,
        load_in_8bit=settings.load_in_8bit,
        verbose=settings.load_in_8bit,
    )

    def set_settings(_settings: Settings):
        global settings
        settings = _settings

    set_settings(settings)
    return app


llama_outer_lock = Lock()
llama_inner_lock = Lock()


def get_llama():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield llama2
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


def get_settings():
    yield settings


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
                if settings.interrupt_requests and llama_outer_lock.locked():
                    await inner_send_chan.send(dict(data="[DONE]"))
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e


stream_field = Field(
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)
max_new_tokens_field = Field(
    default=1000, ge=1, description="The maximum number of tokens to generate."
)

temperature_field = Field(
    default=0.9,
    ge=0.0,
    le=2.0,
    description="The temperature to use for sampling.",
)

top_p_field = Field(
    default=1.0,
    ge=0.0,
    le=1.0,
    description="The top-p value to use for sampling.",
)
top_k_field = Field(
    default=40,
    ge=0,
    description="The top-k value to use for sampling.",
)
repetition_penalty_field = Field(
    default=1.0,
    ge=0.0,
    description="The penalty to apply to repeated tokens.",
)
# stop_field = Field(
#     default=None,
#     description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
# )


class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate text from."
    )
    stream: bool = stream_field
    max_new_tokens: int = max_new_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    top_k: int = top_k_field
    repetition_penalty: float = repetition_penalty_field
    # stop: Optional[Union[str, List[str]]] = stop_field

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                    # "stop": ["\n", "###"],
                }
            ]
        }
    }


@router.post(
    "/v1/completions",
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    llama2: LLAMA2_WRAPPER = Depends(get_llama),
) -> Completion:
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    kwargs = body.model_dump()

    iterator_or_completion: Union[
        Completion, Iterator[CompletionChunk]
    ] = await run_in_threadpool(llama2.completion, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[CompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion


class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="The role of the message."
    )
    content: str = Field(default="", description="The content of the message.")


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    stream: bool = stream_field
    max_new_tokens: int = max_new_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    top_k: int = top_k_field
    repetition_penalty: float = repetition_penalty_field
    # stop: Optional[List[str]] = stop_field

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        ChatCompletionRequestMessage(
                            role="system", content="You are a helpful assistant."
                        ).model_dump(),
                        ChatCompletionRequestMessage(
                            role="user", content="What is the capital of France?"
                        ).model_dump(),
                    ]
                }
            ]
        }
    }


@router.post(
    "/v1/chat/completions",
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    llama2: LLAMA2_WRAPPER = Depends(get_llama),
    settings: Settings = Depends(get_settings),
) -> ChatCompletion:
    kwargs = body.model_dump()

    iterator_or_completion: Union[
        ChatCompletion, Iterator[ChatCompletionChunk]
    ] = await run_in_threadpool(llama2.chat_completion, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[ChatCompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


@router.get("/v1/models")
async def get_models(
    settings: Settings = Depends(get_settings),
) -> ModelList:
    assert llama2 is not None

    return {
        "object": "list",
        "data": [
            {
                "id": settings.backend_type + " default model"
                if settings.model_path == ""
                else settings.model_path,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
        ],
    }
