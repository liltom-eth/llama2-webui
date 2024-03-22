import os
import argparse
from typing import Iterator

import gradio as gr
from dotenv import load_dotenv
from distutils.util import strtobool

from llama2_wrapper import LLAMA2_WRAPPER

import logging

from prompts.utils import PromtsContainer


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
        "--gptq_gpu_memory",
        type=str,
        default="",
        help="Set GPU maximum memory for GPTQ backend to use multiple GPUs, "
             "e.g. \"0:23GiB,1:23GiB\"",
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
    parser.add_argument(
        "--listen",
        type=bool,
        default=False,
        help="listen on 0.0.0.0, allowing to respond to network requests",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port to listen on, default 7860",
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
    GPTQ_GPU_MEMORY = args.gptq_gpu_memory

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
        gptq_gpu_memory=GPTQ_GPU_MEMORY,
        # verbose=True,
    )

    DESCRIPTION = """
    # llama2-webui
    """
    DESCRIPTION2 = """
    - Supporting models: [Llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)/[13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)/[70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf), [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML), [CodeLlama](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GPTQ) ...
    - Supporting model backends: [tranformers](https://github.com/huggingface/transformers), [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ), [llama.cpp](https://github.com/ggerganov/llama.cpp)
    """

    def clear_and_save_textbox(message: str) -> tuple[str, str]:
        return "", message

    def save_textbox_for_prompt(message: str) -> str:
        logging.info("start save_textbox_from_prompt")
        message = convert_summary_to_prompt(message)
        return message

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
        try:
            history = history_with_input[:-1]
            generator = llama2_wrapper.run(
                message,
                history,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            )
            try:
                first_response = next(generator)
                yield history + [(message, first_response)]
            except StopIteration:
                yield history + [(message, "")]
            for response in generator:
                yield history + [(message, response)]
        except Exception as e:
            logging.exception(e)

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

    prompts_container = PromtsContainer()
    prompts = prompts_container.get_prompts_tab_dict()
    default_prompts_checkbox = False
    default_advanced_checkbox = False

    def convert_summary_to_prompt(summary):
        return prompts_container.get_prompt_by_summary(summary)

    def two_columns_list(tab_data, chatbot):
        result = []
        for i in range(int(len(tab_data) / 2) + 1):
            row = gr.Row()
            with row:
                for j in range(2):
                    index = 2 * i + j
                    if index >= len(tab_data):
                        break
                    item = tab_data[index]
                    with gr.Group():
                        gr.HTML(
                            f'<p style="color: black; font-weight: bold;">{item["act"]}</p>'
                        )
                        prompt_text = gr.Button(
                            label="",
                            value=f"{item['summary']}",
                            size="sm",
                            elem_classes="text-left-aligned",
                        )
                        prompt_text.click(
                            fn=save_textbox_for_prompt,
                            inputs=prompt_text,
                            outputs=saved_input,
                            api_name=False,
                            queue=True,
                        ).then(
                            fn=display_input,
                            inputs=[saved_input, chatbot],
                            outputs=chatbot,
                            api_name=False,
                            queue=True,
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
                result.append(row)
        return result

    CSS = """
        .contain { display: flex; flex-direction: column;}
        #component-0 #component-1 #component-2 #component-4 #component-5 { height:71vh !important; }
        #component-0 #component-1 #component-24 > div:nth-child(2) { height:80vh !important; overflow-y:auto }
        .text-left-aligned {text-align: left !important; font-size: 16px;}
    """
    with gr.Blocks(css=CSS) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
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
                with gr.Row():
                    advanced_checkbox = gr.Checkbox(
                        label="Advanced",
                        value=default_prompts_checkbox,
                        container=False,
                        elem_classes="min_check",
                    )
                    prompts_checkbox = gr.Checkbox(
                        label="Prompts",
                        value=default_prompts_checkbox,
                        container=False,
                        elem_classes="min_check",
                    )
                with gr.Column(visible=default_advanced_checkbox) as advanced_column:
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
            with gr.Column(scale=1, visible=default_prompts_checkbox) as prompt_column:
                gr.HTML(
                    '<p style="color: green; font-weight: bold;font-size: 16px;">\N{four leaf clover} prompts</p>'
                )
                for k, v in prompts.items():
                    with gr.Tab(k, scroll_to_output=True):
                        lst = two_columns_list(v, chatbot)
            prompts_checkbox.change(
                lambda x: gr.update(visible=x),
                prompts_checkbox,
                prompt_column,
                queue=False,
            )
            advanced_checkbox.change(
                lambda x: gr.update(visible=x),
                advanced_checkbox,
                advanced_column,
                queue=False,
            )

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

    launch_params = {
        "share": args.share,
        "server_port": args.port
    }

    if args.listen:
        launch_params["server_name"] = "0.0.0.0"

    demo.queue(max_size=20).launch(**launch_params)


if __name__ == "__main__":
    main()
