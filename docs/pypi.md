# llama2-wrapper

Use [llama2-wrapper](https://pypi.org/project/llama2-wrapper/) as your local llama2 backend for Generative Agents/Apps, [colab example](https://github.com/liltom-eth/llama2-webui/blob/main/colab/Llama_2_7b_Chat_GPTQ.ipynb). 

## Features

- Supporting models: [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)/[13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)/[70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf), all [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), all [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) ...
- Supporting model backends: [tranformers](https://github.com/huggingface/transformers), [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ), [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Demos: [Run Llama2 on MacBook Air](https://twitter.com/liltom_eth/status/1682791729207070720?s=20); [Run Llama2 on Colab T4 GPU](https://github.com/liltom-eth/llama2-webui/blob/main/colab/Llama_2_7b_Chat_GPTQ.ipynb)
- [News](https://github.com/liltom-eth/llama2-webui/blob/main/docs/news.md), [Benchmark](https://github.com/liltom-eth/llama2-webui/blob/main/docs/performance.md), [Issue Solutions](https://github.com/liltom-eth/llama2-webui/blob/main/docs/issues.md)

[llama2-wrapper](https://pypi.org/project/llama2-wrapper/)  is the backend and part of [llama2-webui](https://github.com/liltom-eth/llama2-webui), which can run any Llama 2 locally with gradio UI on GPU or CPU from anywhere (Linux/Windows/Mac).

## Install

```bash
pip install llama2-wrapper
```

## Usage

Use `llama2-wrapper` to run ggml llama2 model on CPU, [colab example](https://github.com/liltom-eth/llama2-webui/blob/main/colab/ggmlv3_q4_0.ipynb):

```python
from llama2_wrapper import LLAMA2_WRAPPER, get_prompt 
llama2_wrapper = LLAMA2_WRAPPER()
# Default running on backend llama.cpp.
# Automatically downloading model to: ./models/llama-2-7b-chat.ggmlv3.q4_0.bin
prompt = "Do you know Pytorch"
answer = llama2_wrapper(get_prompt(prompt), temperature=0.9)
```

Run gptq llama2 model on Nvidia GPU, [colab example](https://github.com/liltom-eth/llama2-webui/blob/main/colab/Llama_2_7b_Chat_GPTQ.ipynb):

```python
from llama2_wrapper import LLAMA2_WRAPPER 
llama2_wrapper = LLAMA2_WRAPPER(backend_type="gptq")
# Automatically downloading model to: ./models/Llama-2-7b-Chat-GPTQ
```

Run llama2 7b with bitsandbytes 8 bit with a `model_path`:

```python
from llama2_wrapper import LLAMA2_WRAPPER 
llama2_wrapper = LLAMA2_WRAPPER(
	model_path = "./models/Llama-2-7b-chat-hf",
  backend_type = "transformers",
  load_in_8bit = True
)
```