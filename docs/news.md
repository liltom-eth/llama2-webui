# News
- [2023/08] 🔥 For developers, we offer a web server that acts as a drop-in replacement for the OpenAI API.

  - Usage: 

    ```
    python3 -m llama2_wrapper.server
    ```



- [2023/08] 🔥 For developers, we released `llama2-wrapper`  as a llama2 backend wrapper in [PYPI](https://pypi.org/project/llama2-wrapper/).

  - Install: `pip install llama2-wrapper`

  - Usage: 

    ```python
    from llama2_wrapper import LLAMA2_WRAPPER, get_prompt 
    llama2_wrapper = LLAMA2_WRAPPER(
        model_path="./models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q4_0.bin",
        backend_type="llama.cpp", #options: llama.cpp, transformers, gptq
    )
    prompt = "Do you know Pytorch"
    llama2_promt = get_prompt(prompt)
    answer = llama2_wrapper(llama2_promt, temperature=0.9)
    ```

- [2023/08] 🔥 We added `benchmark.py` for users to benchmark llama2 models on their local devices.

  - Check/contribute the performance of your device in the full [performance doc](https://github.com/liltom-eth/llama2-webui/blob/main/docs/performance.md).

- [2023/07] We released **[llama2-webui](https://github.com/liltom-eth/llama2-webui)**, a gradio web UI to run Llama 2 on GPU or CPU from anywhere (Linux/Windows/Mac). 

  - Supporting models: [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)/[13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)/[70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf), all [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), all [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) ...
  - Supporting model backends:  [tranformers](https://github.com/huggingface/transformers), [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ), [llama.cpp](https://github.com/ggerganov/llama.cpp)