from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class LLAMA2_WRAPPER:
    def __init__(self, config: dict = {}):
        self.config = config
        self.model = None
        self.tokenizer = None

    def init_model(self):
        load_in_8bit = self.config.get("load_in_8bit", True)
        load_in_4bit = self.config.get("load_in_4bit", False)
        if self.model is None:
            self.model = LLAMA2_WRAPPER.create_llama2_model(
                self.config["model_name"],
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        self.model.eval()

    def init_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = LLAMA2_WRAPPER.create_llama2_tokenizer(
                self.config["model_name"]
            )

    @classmethod
    def create_llama2_model(cls, model_name, load_in_8bit=False, load_in_4bit=False):
        if load_in_4bit:
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
            )
        return model

    @classmethod
    def create_llama2_tokenizer(cls, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer

    def get_input_token_length(
        self, message: str, chat_history: list[tuple[str, str]], system_prompt: str
    ) -> int:
        prompt = get_prompt(message, chat_history, system_prompt)
        input_ids = self.tokenizer([prompt], return_tensors="np")["input_ids"]
        return input_ids.shape[-1]

    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Iterator[str]:
        prompt = get_prompt(message, chat_history, system_prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield "".join(outputs)


def get_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    for user_input, response in chat_history:
        texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
    texts.append(f"{message.strip()} [/INST]")
    return "".join(texts)
