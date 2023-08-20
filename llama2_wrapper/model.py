import os
from enum import Enum
from threading import Thread
from typing import Any, Iterator, Union


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

        self.default_llamacpp_path = "./models/llama-2-7b-chat.ggmlv3.q4_0.bin"
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
                        repo_id="TheBloke/Llama-2-7B-Chat-GGML",
                        filename="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        local_dir="./models/",
                    )
                else:
                    print("Model exists in ./models/llama-2-7b-chat.ggmlv3.q4_0.bin.")
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
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
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
                output = self.tokenizer.decode(output_ids[0])
                return output.split("[/INST]")[1].split("</s>")[0]


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
