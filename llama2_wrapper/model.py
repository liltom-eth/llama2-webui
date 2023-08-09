from enum import Enum
from threading import Thread
from typing import Any, Iterator


class LLAMA2_WRAPPER:
    def __init__(
        self,
        model_path: str = None,
        backend_type: str = "llama.cpp",
        max_tokens: int = 4000,
        load_in_8bit: bool = True,
        verbose: bool = False,
    ):
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
        if self.backend_type is BackendType.LLAMA_CPP:
            inputs = self.model.tokenize(bytes(prompt, "utf-8"))

            generator = self.model.generate(
                inputs,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
            outputs = []
            for token in generator:
                if token == self.model.token_eos():
                    break
                b_text = self.model.detokenize([token])
                text = str(b_text, encoding="utf-8", errors="ignore")
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
        prompt = get_prompt(message, chat_history, system_prompt)
        return self.generate(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> str:
        if self.backend_type is BackendType.LLAMA_CPP:
            return self.model.__call__(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                **kwargs,
            )
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            output = self.model.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
            return output, self.tokenizer.decode(output[0])


def get_prompt(
    message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = ""
) -> str:
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
