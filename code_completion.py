import argparse

import gradio as gr
from llama2_wrapper import LLAMA2_WRAPPER

FIM_PREFIX = "<PRE> "
FIM_MIDDLE = " <MID>"
FIM_SUFFIX = " <SUF>"

FIM_INDICATOR = "<FILL_ME>"

EOS_STRING = "</s>"
EOT_STRING = "<EOT>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/codellama-7b-instruct.ggmlv3.Q4_0.bin",
        help="model path",
    )
    parser.add_argument(
        "--backend_type",
        type=str,
        default="llama.cpp",
        help="Backend options: llama.cpp, gptq, transformers",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Maximum context size.",
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

    llama2_wrapper = LLAMA2_WRAPPER(
        model_path=args.model_path,
        backend_type=args.backend_type,
        max_tokens=args.max_tokens,
        load_in_8bit=args.load_in_8bit,
    )

    def generate(
        prompt,
        temperature=0.9,
        max_new_tokens=256,
        top_p=0.95,
        repetition_penalty=1.0,
    ):
        temperature = float(temperature)
        if temperature < 1e-2:
            temperature = 1e-2
        top_p = float(top_p)
        fim_mode = False

        generate_kwargs = dict(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=True,
        )

        if FIM_INDICATOR in prompt:
            fim_mode = True
            try:
                prefix, suffix = prompt.split(FIM_INDICATOR)
            except:
                raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt!")
            prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

        stream = llama2_wrapper.__call__(prompt, **generate_kwargs)

        if fim_mode:
            output = prefix
        else:
            output = prompt

        # for response in stream:
        #     output += response
        #     yield output
        # return output

        previous_token = ""
        for response in stream:
            if any([end_token in response for end_token in [EOS_STRING, EOT_STRING]]):
                if fim_mode:
                    output += suffix
                    yield output
                    return output
                    print("output", output)
                else:
                    return output
            else:
                output += response
            previous_token = response
            yield output
        return output

    examples = [
        'def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\nprint(remove_non_ascii(\'afkdj$$(\'))',
        "X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)\n\n# Train a logistic regression model, predict the labels on the test set and compute the accuracy score",
        "// Returns every other value in the array as a new array.\nfunction everyOther(arr) {",
        "Poor English: She no went to the market. Corrected English:",
        "def alternating(list1, list2):\n   results = []\n   for i in range(min(len(list1), len(list2))):\n       results.append(list1[i])\n       results.append(list2[i])\n   if len(list1) > len(list2):\n       <FILL_ME>\n   else:\n       results.extend(list2[i+1:])\n   return results",
    ]

    def process_example(args):
        for x in generate(args):
            pass
        return x

    description = """
    <div style="text-align: center;">
        <h1>Code Llama Playground</h1>
    
    </div>
    <div style="text-align: center;">
        <p>This is a demo to complete code with Code Llama. For instruction purposes, please use llama2-webui app.py with CodeLlama-Instruct models. </p>
    </div>
    """
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown(description)
            with gr.Row():
                with gr.Column():
                    instruction = gr.Textbox(
                        placeholder="Enter your code here",
                        lines=5,
                        label="Input",
                        elem_id="q-input",
                    )
                    submit = gr.Button("Generate", variant="primary")
                    output = gr.Code(elem_id="q-output", lines=30, label="Output")
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion("Advanced settings", open=False):
                                with gr.Row():
                                    column_1, column_2 = gr.Column(), gr.Column()
                                    with column_1:
                                        temperature = gr.Slider(
                                            label="Temperature",
                                            value=0.1,
                                            minimum=0.0,
                                            maximum=1.0,
                                            step=0.05,
                                            interactive=True,
                                            info="Higher values produce more diverse outputs",
                                        )
                                        max_new_tokens = gr.Slider(
                                            label="Max new tokens",
                                            value=256,
                                            minimum=0,
                                            maximum=8192,
                                            step=64,
                                            interactive=True,
                                            info="The maximum numbers of new tokens",
                                        )
                                    with column_2:
                                        top_p = gr.Slider(
                                            label="Top-p (nucleus sampling)",
                                            value=0.90,
                                            minimum=0.0,
                                            maximum=1,
                                            step=0.05,
                                            interactive=True,
                                            info="Higher values sample more low-probability tokens",
                                        )
                                        repetition_penalty = gr.Slider(
                                            label="Repetition penalty",
                                            value=1.05,
                                            minimum=1.0,
                                            maximum=2.0,
                                            step=0.05,
                                            interactive=True,
                                            info="Penalize repeated tokens",
                                        )

                    gr.Examples(
                        examples=examples,
                        inputs=[instruction],
                        cache_examples=False,
                        fn=process_example,
                        outputs=[output],
                    )

        submit.click(
            generate,
            inputs=[
                instruction,
                temperature,
                max_new_tokens,
                top_p,
                repetition_penalty,
            ],
            outputs=[output],
        )
    demo.queue(concurrency_count=16).launch(share=args.share)


if __name__ == "__main__":
    main()
