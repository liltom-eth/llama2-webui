import os
import time
import argparse

from dotenv import load_dotenv
from distutils.util import strtobool
from llama2_wrapper import LLAMA2_WRAPPER
from memory_profiler import memory_usage
from tqdm import tqdm

def run_iteration(llama2_wrapper, prompt_example, DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_NEW_TOKENS):
    def generation():
        generator = llama2_wrapper.run(
            prompt_example, [], DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_NEW_TOKENS, 1, 0.95, 50
        )
        model_response = None
        try:
            first_model_response = next(generator)
        except StopIteration:
            pass
        for model_response in generator:
            pass
        return llama2_wrapper.get_token_length(model_response), model_response

    tic = time.perf_counter()
    mem_usage, (output_token_length, model_response) = memory_usage((generation, ), max_usage=True, retval=True)
    toc = time.perf_counter()

    generation_time = toc - tic
    tokens_per_second = output_token_length / generation_time

    return generation_time, tokens_per_second, mem_usage, model_response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=5, help="Number of iterations")
    args = parser.parse_args()

    load_dotenv()

    DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "")
    MAX_MAX_NEW_TOKENS = int(os.getenv("MAX_MAX_NEW_TOKENS", 2048))
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024))
    MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", 4000))

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

    # Initialization
    init_tic = time.perf_counter()
    llama2_wrapper = LLAMA2_WRAPPER(config)
    llama2_wrapper.init_tokenizer()
    llama2_wrapper.init_model()
    init_toc = time.perf_counter()
    initialization_time = init_toc - init_tic

    total_time = 0
    total_tokens_per_second = 0
    total_memory_gen = 0

    prompt_example = "Can you explain briefly to me what is the Python programming language?"

    # Cold run
    print("Performing cold run...")
    run_iteration(llama2_wrapper, prompt_example, DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_NEW_TOKENS)

    # Timed runs
    print(f"Performing {args.iter} timed runs...")
    for _ in tqdm(range(args.iter)):
        gen_time, tokens_per_sec, mem_gen, model_response = run_iteration(llama2_wrapper, prompt_example, DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_NEW_TOKENS)
        total_time += gen_time
        total_tokens_per_second += tokens_per_sec
        total_memory_gen += mem_gen
    avg_time = total_time / args.iter
    avg_tokens_per_second = total_tokens_per_second / args.iter
    avg_memory_gen = total_memory_gen / args.iter

    print(f"Last model response: {model_response}")
    print(f"Initialization time: {initialization_time:0.4f} seconds.")
    print(f"Average generation time over {args.iter} iterations: {avg_time:0.4f} seconds.")
    print(f"Average speed over {args.iter} iterations: {avg_tokens_per_second:0.4f} tokens/sec.")
    print(f"Average memory usage during generation: {avg_memory_gen:.2f} MiB")

if __name__ == "__main__":
    main()


