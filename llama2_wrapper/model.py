import os
import time
import uuid
from enum import Enum
from threading import Thread
from typing import Any, Iterator, Union, List
from llama2_wrapper.types import (
    Completion,
    CompletionChunk,
    ChatCompletion,
    ChatCompletionChunk,
    # ChatCompletionMessage,
    Message,
    B_INST,
    E_INST,
    B_SYS,
    E_SYS,
)


class LLAMA2_WRAPPER:
    def __init__(
        self,
        model_path: str = "",
        backend_type: str = "llama.cpp",
        max_tokens: int = 4000,
        load_in_8bit: bool = True,
        verbose: bool = False,
    ):
        """Load a llama2 model from `model_path`.

        Args:
            model_path: Path to the model.
            backend_type: Backend for llama2, options: llama.cpp, gptq, transformers
            max_tokens: Maximum context size.
            load_in_8bit: Use bitsandbytes to run model in 8 bit mode (only for transformers models).
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A LLAMA2_WRAPPER instance.
        """
        self.model_path = model_path
        self.backend_type = BackendType.get_type(backend_type)
        self.max_tokens = max_tokens
        self.load_in_8bit = load_in_8bit

        self.model = None
        self.tokenizer = None

        self.verbose = verbose

        if self.backend_type is BackendType.LLAMA_CPP:
            print("Running on backend llama.cpp.")
        else:
            import torch

            if torch.cuda.is_available():
                print("Running on GPU with backend torch transformers.")
            else:
                print("GPU CUDA not found.")

        self.default_llamacpp_path = "./models/llama-2-7b-chat.Q4_0.gguf"
        self.default_gptq_path = "./models/Llama-2-7b-Chat-GPTQ"
        # Download default ggml/gptq model
        if self.model_path == "":
            print("Model path is empty.")
            if self.backend_type is BackendType.LLAMA_CPP:
                print("Use default llama.cpp model path: " + self.default_llamacpp_path)
                if not os.path.exists(self.default_llamacpp_path):
                    print("Start downloading model to: " + self.default_llamacpp_path)
                    from huggingface_hub import hf_hub_download

                    hf_hub_download(
                        repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
                        filename="llama-2-7b-chat.Q4_0.gguf",
                        local_dir="./models/",
                    )
                else:
                    print("Model exists in ./models/llama-2-7b-chat.Q4_0.gguf.")
                self.model_path = self.default_llamacpp_path
            elif self.backend_type is BackendType.GPTQ:
                print("Use default gptq model path: " + self.default_gptq_path)
                if not os.path.exists(self.default_gptq_path):
                    print("Start downloading model to: " + self.default_gptq_path)
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        "TheBloke/Llama-2-7b-Chat-GPTQ",
                        local_dir=self.default_gptq_path,
                    )
                else:
                    print("Model exists in " + self.default_gptq_path)
                self.model_path = self.default_gptq_path

        self.init_tokenizer()
        self.init_model()

    def init_model(self):
        if self.model is None:
            self.model = LLAMA2_WRAPPER.create_llama2_model(
                self.model_path,
                self.backend_type,
                self.max_tokens,
                self.load_in_8bit,
                self.verbose,
            )
        if self.backend_type is not BackendType.LLAMA_CPP:
            self.model.eval()

    def init_tokenizer(self):
        if self.backend_type is not BackendType.LLAMA_CPP:
            if self.tokenizer is None:
                self.tokenizer = LLAMA2_WRAPPER.create_llama2_tokenizer(self.model_path)

    @classmethod
    def create_llama2_model(
        cls, model_path, backend_type, max_tokens, load_in_8bit, verbose
    ):
        if backend_type is BackendType.LLAMA_CPP:
            from llama_cpp import Llama

            model = Llama(
                model_path=model_path,
                n_ctx=max_tokens,
                n_batch=max_tokens,
                verbose=verbose,
            )
        elif backend_type is BackendType.GPTQ:
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
        elif backend_type is BackendType.TRANSFORMERS:
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
            )
        else:
            print(backend_type + "not implemented.")
        return model

    @classmethod
    def create_llama2_tokenizer(cls, model_path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def get_token_length(
        self,
        prompt: str,
    ) -> int:
        if self.backend_type is BackendType.LLAMA_CPP:
            input_ids = self.model.tokenize(bytes(prompt, "utf-8"))
            return len(input_ids)
        else:
            input_ids = self.tokenizer([prompt], return_tensors="np")["input_ids"]
            return input_ids.shape[-1]

    def get_input_token_length(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
    ) -> int:
        prompt = get_prompt(message, chat_history, system_prompt)

        return self.get_token_length(prompt)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Create a generator of response from a prompt.

        Examples:
            >>> llama2_wrapper = LLAMA2_WRAPPER()
            >>> prompt = get_prompt("Hi do you know Pytorch?")
            >>> for response in llama2_wrapper.generate(prompt):
            ...     print(response)

        Args:
            prompt: The prompt to generate text from.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        if self.backend_type is BackendType.LLAMA_CPP:
            result = self.model(
                prompt=prompt,
                stream=True,
                max_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            outputs = []
            for part in result:
                text = part["choices"][0]["text"]
                outputs.append(text)
                yield "".join(outputs)
        else:
            from transformers import TextIteratorStreamer

            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

            outputs = []
            for text in streamer:
                outputs.append(text)
                yield "".join(outputs)

    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
    ) -> Iterator[str]:
        """Create a generator of response from a chat message.
        Process message to llama2 prompt with chat history
        and system_prompt for chatbot.

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        prompt = get_prompt(message, chat_history, system_prompt)
        return self.generate(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

    def __call__(
        self,
        prompt: str,
        stream: bool = False,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Generate text from a prompt.

        Examples:
            >>> llama2_wrapper = LLAMA2_WRAPPER()
            >>> prompt = get_prompt("Hi do you know Pytorch?")
            >>> print(llama2_wrapper(prompt))

        Args:
            prompt: The prompt to generate text from.
            stream: Whether to stream the results.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Generated text.
        """
        if self.backend_type is BackendType.LLAMA_CPP:
            completion_or_chunks = self.model.__call__(
                prompt,
                stream=stream,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            if stream:

                def chunk_generator(chunks):
                    for part in chunks:
                        chunk = part["choices"][0]["text"]
                        yield chunk

                chunks: Iterator[str] = chunk_generator(completion_or_chunks)
                return chunks
            return completion_or_chunks["choices"][0]["text"]
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids
            prompt_tokens_len = len(inputs[0])
            inputs = inputs.to("cuda")
            generate_kwargs = dict(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            if stream:
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=10.0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generate_kwargs["streamer"] = streamer

                t = Thread(target=self.model.generate, kwargs=generate_kwargs)
                t.start()
                return streamer
            else:
                output_ids = self.model.generate(
                    **generate_kwargs,
                )
                # skip prompt, skip special tokens
                output = self.tokenizer.decode(
                    output_ids[0][prompt_tokens_len:], skip_special_tokens=True
                )
                return output

    def completion(
        self,
        prompt: str,
        stream: bool = False,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """For OpenAI compatible API /v1/completions
        Generate text from a prompt.

        Examples:
            >>> llama2_wrapper = LLAMA2_WRAPPER()
            >>> prompt = get_prompt("Hi do you know Pytorch?")
            >>> print(llm.completion(prompt))

        Args:
            prompt: The prompt to generate text from.
            stream: Whether to stream the results.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = (
            self.backend_type + " default model"
            if self.model_path == ""
            else self.model_path
        )
        if self.backend_type is BackendType.LLAMA_CPP:
            completion_or_chunks = self.model.__call__(
                prompt,
                stream=stream,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            if stream:
                chunks: Iterator[CompletionChunk] = completion_or_chunks
                return chunks
            return completion_or_chunks
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids
            prompt_tokens_len = len(inputs[0])
            inputs = inputs.to("cuda")
            generate_kwargs = dict(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            if stream:
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=10.0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generate_kwargs["streamer"] = streamer

                t = Thread(target=self.model.generate, kwargs=generate_kwargs)
                t.start()

                def chunk_generator(chunks):
                    for part in chunks:
                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": part,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }

                chunks: Iterator[CompletionChunk] = chunk_generator(streamer)
                return chunks

            else:
                output_ids = self.model.generate(
                    **generate_kwargs,
                )
                total_tokens_len = len(output_ids[0])
                output = self.tokenizer.decode(
                    output_ids[0][prompt_tokens_len:], skip_special_tokens=True
                )
                completion: Completion = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": output,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens_len,
                        "completion_tokens": total_tokens_len - prompt_tokens_len,
                        "total_tokens": total_tokens_len,
                    },
                }
                return completion

    def chat_completion(
        self,
        messages: List[Message],
        stream: bool = False,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """For OpenAI compatible API /v1/chat/completions
        Generate text from a dialog (chat history).

        Examples:
            >>> llama2_wrapper = LLAMA2_WRAPPER()
            >>> dialog = [
                    {
                        "role":"system",
                        "content":"You are a helpful, respectful and honest assistant. "
                    },{
                        "role":"user",
                        "content":"Hi do you know Pytorch?",
                    },
                ]
            >>> print(llm.chat_completion(dialog))

        Args:
            dialog: The dialog (chat history) to generate text from.
            stream: Whether to stream the results.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = (
            self.backend_type + " default model"
            if self.model_path == ""
            else self.model_path
        )
        if self.backend_type is BackendType.LLAMA_CPP:
            completion_or_chunks = self.model.create_chat_completion(
                messages,
                stream=stream,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            if stream:
                chunks: Iterator[ChatCompletionChunk] = completion_or_chunks
                return chunks
            return completion_or_chunks
        else:
            prompt = get_prompt_for_dialog(messages)
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids
            prompt_tokens_len = len(inputs[0])
            inputs = inputs.to("cuda")
            generate_kwargs = dict(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # num_beams=1,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            if stream:
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=10.0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generate_kwargs["streamer"] = streamer
                t = Thread(target=self.model.generate, kwargs=generate_kwargs)
                t.start()

                def chunk_generator(chunks):
                    yield {
                        "id": "chat" + completion_id,
                        "model": model_name,
                        "created": created,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    for part in enumerate(chunks):
                        yield {
                            "id": "chat" + completion_id,
                            "model": model_name,
                            "created": created,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": part,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                chunks: Iterator[ChatCompletionChunk] = chunk_generator(streamer)
                return chunks

            else:
                output_ids = self.model.generate(
                    **generate_kwargs,
                )
                total_tokens_len = len(output_ids[0])
                output = self.tokenizer.decode(
                    output_ids[0][prompt_tokens_len:], skip_special_tokens=True
                )
                chatcompletion: ChatCompletion = {
                    "id": "chat" + completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output,
                            },
                            "finish_reason": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens_len,
                        "completion_tokens": total_tokens_len - prompt_tokens_len,
                        "total_tokens": total_tokens_len,
                    },
                }
                return chatcompletion


def get_prompt_for_dialog(dialog: List[Message]) -> str:
    """Process dialog (chat history) to llama2 prompt for
    OpenAI compatible API /v1/chat/completions.

    Examples:
        >>> dialog = [
                {
                    "role":"system",
                    "content":"You are a helpful, respectful and honest assistant. "
                },{
                    "role":"user",
                    "content":"Hi do you know Pytorch?",
                },
            ]
        >>> prompt = get_prompt_for_dialog("Hi do you know Pytorch?")

    Args:
        dialog: The dialog (chat history) to generate text from.

    Yields:
        prompt string.
    """
    # add "<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" in first dialog
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]
    # check roles
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    # add chat history
    texts = []
    for prompt, answer in zip(
        dialog[::2],
        dialog[1::2],
    ):
        texts.append(
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
        )
    # check last message if role is user, then add it to prompt text
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    texts.append(f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}")
    return "".join(texts)


def get_prompt(
    message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = ""
) -> str:
    """Process message to llama2 prompt with chat history
    and system_prompt for chatbot.

    Examples:
        >>> prompt = get_prompt("Hi do you know Pytorch?")

    Args:
        message: The origianl chat message to generate text from.
        chat_history: Chat history list from chatbot.
        system_prompt: System prompt for chatbot.

    Yields:
        prompt string.
    """
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    for user_input, response in chat_history:
        texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
    texts.append(f"{message.strip()} [/INST]")
    return "".join(texts)


class BackendType(Enum):
    UNKNOWN = 0
    TRANSFORMERS = 1
    GPTQ = 2
    LLAMA_CPP = 3

    @classmethod
    def get_type(cls, backend_name: str):
        backend_type = None
        backend_name_lower = backend_name.lower()
        if "transformers" in backend_name_lower:
            backend_type = BackendType.TRANSFORMERS
        elif "gptq" in backend_name_lower:
            backend_type = BackendType.GPTQ
        elif "cpp" in backend_name_lower:
            backend_type = BackendType.LLAMA_CPP
        else:
            raise Exception("Unknown backend: " + backend_name)
            # backend_type = BackendType.UNKNOWN
        return backend_type
