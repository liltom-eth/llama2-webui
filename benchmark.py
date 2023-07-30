import os
import time

from dotenv import load_dotenv
from distutils.util import strtobool

from llama2_wrapper import LLAMA2_WRAPPER


def main():
    load_dotenv()

    DEFAULT_SYSTEM_PROMPT = (
        os.getenv("DEFAULT_SYSTEM_PROMPT")
        if os.getenv("DEFAULT_SYSTEM_PROMPT") is not None
        else ""
    )
    MAX_MAX_NEW_TOKENS = (
        int(os.getenv("MAX_MAX_NEW_TOKENS"))
        if os.getenv("DEFAULT_MAX_NEW_TOKENS") is not None
        else 2048
    )
    DEFAULT_MAX_NEW_TOKENS = (
        int(os.getenv("DEFAULT_MAX_NEW_TOKENS"))
        if os.getenv("DEFAULT_MAX_NEW_TOKENS") is not None
        else 1024
    )
    MAX_INPUT_TOKEN_LENGTH = (
        int(os.getenv("MAX_INPUT_TOKEN_LENGTH"))
        if os.getenv("MAX_INPUT_TOKEN_LENGTH") is not None
        else 4000
    )

    MODEL_PATH = os.getenv("MODEL_PATH")
    assert MODEL_PATH is not None, f"MODEL_PATH is required, got: {MODEL_PATH}"

    LOAD_IN_8BIT = bool(strtobool(os.getenv("LOAD_IN_8BIT", "True")))

    LOAD_IN_4BIT = bool(strtobool(os.getenv("LOAD_IN_4BIT", "True")))

    LLAMA_CPP = bool(strtobool(os.getenv("LLAMA_CPP", "True")))

    if LLAMA_CPP:
        print("Running on CPU with llama.cpp.")
    else:
        import torch

        if torch.cuda.is_available():
            print("Running on GPU with torch transformers.")
        else:
            print("CUDA not found.")

    config = {
        "model_name": MODEL_PATH,
        "load_in_8bit": LOAD_IN_8BIT,
        "load_in_4bit": LOAD_IN_4BIT,
        "llama_cpp": LLAMA_CPP,
        "MAX_INPUT_TOKEN_LENGTH": MAX_INPUT_TOKEN_LENGTH,
    }

    tic = time.perf_counter()
    llama2_wrapper = LLAMA2_WRAPPER(config)
    llama2_wrapper.init_tokenizer()
    llama2_wrapper.init_model()
    toc = time.perf_counter()
    print(f"Initialize the model in {toc - tic:0.4f} seconds.")

    example = "Can you explain briefly to me what is the Python programming language?"

    generator = llama2_wrapper.run(
        example, [], DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_NEW_TOKENS, 1, 0.95, 50
    )
    tic = time.perf_counter()
    try:
        first_response = next(generator)
        # history += [(example, first_response)]
    except StopIteration:
        pass
        # history += [(example, "")]
    for response in generator:
        # history += [(example, response)]
        pass
    print(response)

    toc = time.perf_counter()
    output_token_length = llama2_wrapper.get_token_length(response)

    print(f"Generating the out in {toc - tic:0.4f} seconds.")
    print(f"Speed: {output_token_length / (toc - tic):0.4f} tokens/sec.")


if __name__ == "__main__":
    main()
