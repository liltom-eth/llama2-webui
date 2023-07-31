# llama2-webui

Running Llama 2 with gradio web UI on GPU or CPU from anywhere (Linux/Windows/Mac). 
- Supporting all Llama 2 models (7B, 13B, 70B, GPTQ, GGML) with 8-bit, 4-bit mode. 
- Supporting GPU inference with at least 6 GB VRAM, and CPU inference.

![screenshot](./static/screenshot.png)

## Features

- Supporting models: [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)/[13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)/[70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf), all [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), all [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) ...
- Supporting model backends
  - Nvidia GPU: [tranformers](https://github.com/huggingface/transformers), [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ)
    - GPU inference with at least 6 GB VRAM

  - CPU, Mac/AMD GPU: [llama.cpp](https://github.com/ggerganov/llama.cpp)
    - CPU inference [Demo](https://twitter.com/liltom_eth/status/1682791729207070720?s=20) on Macbook Air.

- Web UI interface: gradio 

- [News](./docs/news.md), [Benchmark](./docs/performance.md), [Issue Solutions](./docs/issues.md)

## Contents

- [Install](#install)
- [Download Llama-2 Models](#download-llama-2-models)
  - [Model List](#model-list)
  - [Download Script](#download-script)
- [Usage](#usage)
  - [Config Examples](#config-examples)
  - [Start Web UI](#start-web-ui)
  - [Run on Nvidia GPU](#run-on-nvidia-gpu)
    - [Run on Low Memory GPU with 8 bit](#run-on-low-memory-gpu-with-8-bit)
    - [Run on Low Memory GPU with 4 bit](#run-on-low-memory-gpu-with-4-bit)
  - [Run on CPU](#run-on-cpu)
    - [Mac Metal Acceleration](#mac-metal-acceleration)
    - [AMD/Nvidia GPU Acceleration](#amdnvidia-gpu-acceleration)
  - [Benchmark](#benchmark)
- [License](#license)
- [Contributing](#contributing)
  


## Install
### Method 1: From [PyPI](https://pypi.org/project/llama2-wrapper/)
```
pip install llama2-wrapper
```
### Method 2: From Source:
```
git clone https://github.com/liltom-eth/llama2-webui.git
cd llama2-webui
pip install -r requirements.txt
```
### Install Issues:
`bitsandbytes >= 0.39` may not work on older NVIDIA GPUs. In that case, to use `LOAD_IN_8BIT`, you may have to downgrade like this:

-  `pip install bitsandbytes==0.38.1`

`bitsandbytes` also need a special install for Windows:
```
pip uninstall bitsandbytes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.0-py3-none-win_amd64.whl
```

## Download Llama-2 Models

Llama 2 is a collection of pre-trained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Llama-2-7b-Chat-GPTQ is the GPTQ model files for [Meta's Llama 2 7b Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). GPTQ 4-bit Llama-2 model require less GPU VRAM to run it.

### Model List

| Model Name                     | set MODEL_PATH in .env                   | Download URL                                                 |
| ------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| meta-llama/Llama-2-7b-chat-hf  | /path-to/Llama-2-7b-chat-hf              | [Link](https://huggingface.co/llamaste/Llama-2-7b-chat-hf)   |
| meta-llama/Llama-2-13b-chat-hf | /path-to/Llama-2-13b-chat-hf             | [Link](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)  |
| meta-llama/Llama-2-70b-chat-hf | /path-to/Llama-2-70b-chat-hf             | [Link](https://huggingface.co/llamaste/Llama-2-70b-chat-hf)  |
| meta-llama/Llama-2-7b-hf       | /path-to/Llama-2-7b-hf                   | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf)      |
| meta-llama/Llama-2-13b-hf      | /path-to/Llama-2-13b-hf                  | [Link](https://huggingface.co/meta-llama/Llama-2-13b-hf)     |
| meta-llama/Llama-2-70b-hf      | /path-to/Llama-2-70b-hf                  | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf)     |
| TheBloke/Llama-2-7b-Chat-GPTQ  | /path-to/Llama-2-7b-Chat-GPTQ            | [Link](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ) |
| TheBloke/Llama-2-7B-Chat-GGML  | /path-to/llama-2-7b-chat.ggmlv3.q4_0.bin | [Link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) |
| ...                            | ...                                      | ...                                                          |

Running 4-bit model `Llama-2-7b-Chat-GPTQ` needs GPU with 6GB VRAM. 

Running 4-bit model `llama-2-7b-chat.ggmlv3.q4_0.bin` needs CPU with 6GB RAM. There is also a list of other 2, 3, 4, 5, 6, 8-bit GGML models that can be used from [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML).

### Download Script

These models can be downloaded from the link using CMD like:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:meta-llama/Llama-2-7b-chat-hf
```

To download Llama 2 models, you need to request access from [https://ai.meta.com/llama/](https://ai.meta.com/llama/) and also enable access on repos like [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main). Requests will be processed in hours.

For GPTQ models like [TheBloke/Llama-2-7b-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), you can directly download without requesting access.

For GGML models like [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML), you can directly download without requesting access.

## Usage

### Config Examples

Setup your `MODEL_PATH` and model configs in `.env` file. 

There are some examples in `./env_examples/` folder.

| Model Setup                       | Example .env                |
| --------------------------------- | --------------------------- |
| Llama-2-7b-chat-hf 8-bit on GPU   | .env.7b_8bit_example        |
| Llama-2-7b-Chat-GPTQ 4-bit on GPU | .env.7b_gptq_example        |
| Llama-2-7B-Chat-GGML 4bit on CPU  | .env.7b_ggmlv3_q4_0_example |
| Llama-2-13b-chat-hf on GPU        | .env.13b_example            |
| ...                               | ...                         |

### Start Web UI

Run chatbot with web UI:

```
python app.py
```

### Run on Nvidia GPU

The running requires around 14GB of GPU VRAM for Llama-2-7b and 28GB of GPU VRAM for Llama-2-13b. 

If you are running on multiple GPUs, the model will be loaded automatically on GPUs and split the VRAM usage. That allows you to run Llama-2-7b (requires 14GB of GPU VRAM) on a setup like 2 GPUs (11GB VRAM each).

#### Run on Low Memory GPU with 8 bit

If you do not have enough memory,  you can set up your `LOAD_IN_8BIT` as `True` in `.env`. This can reduce memory usage by around half with slightly degraded model quality. It is compatible with the CPU, GPU, and Metal backend.

Llama-2-7b with 8-bit compression can run on a single GPU with 8 GB of VRAM, like an Nvidia RTX 2080Ti, RTX 4080, T4, V100 (16GB).

#### Run on Low Memory GPU with 4 bit

If you want to run 4 bit  Llama-2 model like `Llama-2-7b-Chat-GPTQ`,  you can set up your `LOAD_IN_4BIT` as `True` in `.env` like example `.env.7b_gptq_example`. 

Make sure you have downloaded the 4-bit model from `Llama-2-7b-Chat-GPTQ` and set the `MODEL_PATH` and arguments in `.env` file.

`Llama-2-7b-Chat-GPTQ` can run on a single GPU with 6 GB of VRAM.

### Run on CPU

Run Llama-2 model on CPU requires [llama.cpp](https://github.com/ggerganov/llama.cpp) dependency and [llama.cpp Python Bindings](https://github.com/abetlen/llama-cpp-python), which are already installed. 


Download GGML models like `llama-2-7b-chat.ggmlv3.q4_0.bin` following [Download Llama-2 Models](#download-llama-2-models) section. `llama-2-7b-chat.ggmlv3.q4_0.bin` model requires at least 6 GB RAM to run on CPU.

Set up configs like `.env.7b_ggmlv3_q4_0_example` from `env_examples` as `.env`.

Run web UI `python app.py` .

#### Mac Metal Acceleration

If you would like to use Mac Metal for acceleration, 

```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
```

or check details:

- [MacOS Install with Metal GPU](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md)

#### AMD/Nvidia GPU Acceleration

If you would like to use AMD/Nvidia GPU for acceleration, check this:

- [Installation with OpenBLAS / cuBLAS / CLBlast / Metal](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal)

### Benchmark

Run benchmark script to compute performance on your device:

```bash
python benchmark.py
```

`benchmark.py` will load the same `.env` as `app.py`.

Some benchmark performance:

| Model                | Precision | Device             | GPU VRAM    | Speed (tokens/sec) | load time (s) |
| -------------------- | --------- | ------------------ | ----------- | -------------------- | ------------- |
| Llama-2-7b-chat-hf   | 8bit      | NVIDIA RTX 2080 Ti | 7.7 GB VRAM | 3.76                 | 783.87        |
| Llama-2-7b-Chat-GPTQ | 4 bit     | NVIDIA RTX 2080 Ti | 5.8 GB VRAM | 12.08                | 192.91        |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Apple M2 CPU       | 5.4GB RAM   | 5.28               | 0.20          |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | Apple M2 Metal | 5.4GB RAM | 9.56 | 0.47 |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit | Intel i7-8700 | 4.5 GB RAM | 5.70 | 71.48 |

Check/contribute the performance of your device in the full [performance doc](./docs/performance.md).

## License

MIT - see [MIT License](LICENSE)

This project enables users to adapt it freely for proprietary purposes without any restrictions.

## Contributing

Kindly read our [Contributing Guide](CONTRIBUTING.md) to learn and understand our development process.

### All Contributors

<a href="https://github.com/liltom-eth/llama2-webui/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=liltom-eth/llama2-webui" />
</a>

### Review
<a href='https://github.com/repo-reviews/repo-reviews.github.io/blob/main/create.md' target="_blank"><img alt='Github' src='https://img.shields.io/badge/review-100000?style=flat&logo=Github&logoColor=white&labelColor=888888&color=555555'/></a>

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=liltom-eth/llama2-webui&type=Date)](https://star-history.com/#liltom-eth/llama2-webui&Date)

## Credits

- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat
- https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ
- [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [https://github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
