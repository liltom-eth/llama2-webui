import os
import argparse
from typing import Iterator

import gradio as gr
from dotenv import load_dotenv
from distutils.util import strtobool

from llama2_wrapper import LLAMA2_WRAPPER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument(
        "--backend_type",
        type=str,
        default="",
        help="Backend options: llama.cpp, gptq, transformers",
    )
    parser.add_argument(
        "--load_in_8bit",
        type=bool,
        default=False,
        help="Whether to use bitsandbytes 8 bit.",
    )
    parser.add_argument(
        "--share",
        type=bool,
        default=False,
        help="Whether to share public for gradio.",
    )
    args = parser.parse_args()

    load_dotenv()

    DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "")
    MAX_MAX_NEW_TOKENS = int(os.getenv("MAX_MAX_NEW_TOKENS", 2048))
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 1024))
    MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", 4000))

    MODEL_PATH = os.getenv("MODEL_PATH")
    assert MODEL_PATH is not None, f"MODEL_PATH is required, got: {MODEL_PATH}"
    BACKEND_TYPE = os.getenv("BACKEND_TYPE")
    assert BACKEND_TYPE is not None, f"BACKEND_TYPE is required, got: {BACKEND_TYPE}"

    LOAD_IN_8BIT = bool(strtobool(os.getenv("LOAD_IN_8BIT", "True")))

    if args.model_path != "":
        MODEL_PATH = args.model_path
    if args.backend_type != "":
        BACKEND_TYPE = args.backend_type
    if args.load_in_8bit:
        LOAD_IN_8BIT = True

    llama2_wrapper = LLAMA2_WRAPPER(
        model_path=MODEL_PATH,
        backend_type=BACKEND_TYPE,
        max_tokens=MAX_INPUT_TOKEN_LENGTH,
        load_in_8bit=LOAD_IN_8BIT,
        # verbose=True,
    )

    DESCRIPTION = """
    # llama2-webui

    This is a chatbot based on Llama-2. 
    """
    DESCRIPTION2 = """
    - Supporting models: [Llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)/[13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)/[70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf), [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML), [CodeLlama](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GPTQ) ...
    - Supporting model backends: [tranformers](https://github.com/huggingface/transformers), [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ), [llama.cpp](https://github.com/ggerganov/llama.cpp)
    """

    def clear_and_save_textbox(message: str) -> tuple[str, str]:
        return "", message

    def display_input(
        message: str, history: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        history.append((message, ""))
        return history

    def delete_prev_fn(
        history: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], str]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ""
        return history, message or ""

    def generate(
        message: str,
        history_with_input: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Iterator[list[tuple[str, str]]]:
        if max_new_tokens > MAX_MAX_NEW_TOKENS:
            raise ValueError

        history = history_with_input[:-1]
        generator = llama2_wrapper.run(
            message, history, system_prompt, max_new_tokens, temperature, top_p, top_k
        )
        try:
            first_response = next(generator)
            yield history + [(message, first_response)]
        except StopIteration:
            yield history + [(message, "")]
        for response in generator:
            yield history + [(message, response)]

    def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
        generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
        for x in generator:
            pass
        return "", x

    def check_input_token_length(
        message: str, chat_history: list[tuple[str, str]], system_prompt: str
    ) -> None:
        input_token_length = llama2_wrapper.get_input_token_length(
            message, chat_history, system_prompt
        )
        if input_token_length > MAX_INPUT_TOKEN_LENGTH:
            raise gr.Error(
                f"The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
            )

    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Group():
            chatbot = gr.Chatbot(label="Chatbot")
            with gr.Row():
                textbox = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="Type a message...",
                    scale=10,
                )
                submit_button = gr.Button(
                    "Submit", variant="primary", scale=1, min_width=0
                )
        with gr.Row():
            retry_button = gr.Button("🔄  Retry", variant="secondary")
            undo_button = gr.Button("↩️ Undo", variant="secondary")
            clear_button = gr.Button("🗑️  Clear", variant="secondary")

        saved_input = gr.State()

        with gr.Accordion(label="Advanced options", open=False):
            system_prompt = gr.Textbox(
                label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=6
            )
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=MAX_MAX_NEW_TOKENS,
                step=1,
                value=DEFAULT_MAX_NEW_TOKENS,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=4.0,
                step=0.1,
                value=1.0,
            )
            top_p = gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.95,
            )
            top_k = gr.Slider(
                label="Top-k",
                minimum=1,
                maximum=1000,
                step=1,
                value=50,
            )

        gr.Examples(
            examples=[
                "Hello there! How are you doing?",
                "Can you explain briefly to me what is the Python programming language?",
                "Explain the plot of Cinderella in a sentence.",
                "How many hours does it take a man to eat a Helicopter?",
                "Write a 100-word article on 'Benefits of Open-Source in AI research'",
            ],
            inputs=textbox,
            outputs=[textbox, chatbot],
            fn=process_example,
            cache_examples=True,
        )

        gr.Markdown(DESCRIPTION2)

        textbox.submit(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        ).success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )

        button_event_preprocess = (
            submit_button.click(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
            )
            .then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            )
            .then(
                fn=check_input_token_length,
                inputs=[saved_input, chatbot, system_prompt],
                api_name=False,
                queue=False,
            )
            .success(
                fn=generate,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            )
        )

        retry_button.click(
            fn=delete_prev_fn,
            inputs=chatbot,
            outputs=[chatbot, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).then(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )

        undo_button.click(
            fn=delete_prev_fn,
            inputs=chatbot,
            outputs=[chatbot, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=lambda x: x,
            inputs=[saved_input],
            outputs=textbox,
            api_name=False,
            queue=False,
        )

        clear_button.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, saved_input],
            queue=False,
            api_name=False,
        )

    demo.queue(max_size=20).launch(share=args.share)


if __name__ == "__main__":
    main()
