# llama2-webui

Running Llama 2 with gradio web UI on your desktop GPU. Supporting 8-bit, 4-bit mode on Llama 2 7B, 13B, 70B.

## Features

- Web UI interface: gradio 
- Supporting models: Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-2-7b-Chat-GPTQ
- Supporting GPU run with at least 6 GB VRAM
- Supporting 8-bit mode with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- Supporting 4-bit mode with [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

## Contents

- [Install](#install)
- [Download Llama-2 Models](#download-llama-2-models)
  - [Model List](#model-list)
  - [Download Script](#download-script)
- [Usage](#usage)
  - [Start Web UI](#start-web-ui)
  - [Run on Low Memory GPU with 8 bit](#run-on-low-memory-gpu-with-8-bit)
  - [Run on Low Memory GPU with 4 bit](#run-on-low-memory-gpu-with-4-bit)


## Install

```
pip install -r requirements.txt
```

`bitsandbytes >= 0.39` may not work on older NVIDIA GPUs. In that case, to use `LOAD_IN_8BIT`, you may have to downgrade like this:

-  `pip install bitsandbytes==0.38.1`

## Download Llama-2 Models

Llama 2 is a collection of pre-trained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters.

Llama-2-7b-Chat-GPTQ is the GPTQ model files for [Meta's Llama 2 7b Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). GPTQ 4-bit Llama-2 model require less GPU VRAM to run it.

### Model List

| Model Name                     | set MODEL_PATH in .env        | Download URL                                                 |
| ------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| meta-llama/Llama-2-7b-chat-hf  | /path-to/Llama-2-7b-chat-hf   | [Link](https://huggingface.co/llamaste/Llama-2-7b-chat-hf)   |
| meta-llama/Llama-2-13b-chat-hf | /path-to/Llama-2-13b-chat-hf  | [Link](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)  |
| meta-llama/Llama-2-70b-chat-hf | /path-to/Llama-2-70b-chat-hf  | [Link](https://huggingface.co/llamaste/Llama-2-70b-chat-hf)  |
| meta-llama/Llama-2-7b-hf       | /path-to/Llama-2-7b-hf        | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf)      |
| meta-llama/Llama-2-13b-hf      | /path-to/Llama-2-13b-hf       | [Link](https://huggingface.co/meta-llama/Llama-2-13b-hf)     |
| meta-llama/Llama-2-70b-hf      | /path-to/Llama-2-70b-hf       | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf)     |
| TheBloke/Llama-2-7b-Chat-GPTQ  | /path-to/Llama-2-7b-Chat-GPTQ | [Link](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ) |

### Download Script

These models can be downloaded from the link using CMD like:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@hf.co:meta-llama/Llama-2-7b-chat-hf
```

To download Llama 2 models, you need to request access from [https://ai.meta.com/llama/](https://ai.meta.com/llama/) and also enable access on repos like [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main). Requests will be processed in hours.

For GPTQ models like [TheBloke/Llama-2-7b-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ), you can directly download without requesting access.

## Usage

### Start  Web UI

Setup your `MODEL_PATH` in `.env` file. 

Check `.env.7b_8bit_example` as a reference to run `Llama-2-7b` on 8-bit mode.

Check `.env.7b_gptq_example` as a reference to run `Llama-2-7b-Chat-GPTQ` on 4-bit mode.

Check `.env.13b_example` as a reference to run `Llama-2-13b` without quantization.

```
python app.py
```

### Run on GPU

The running requires around 14GB of GPU VRAM for Llama-2-7b and 28GB of GPU VRAM for Llama-2-13b. 

If you are running on multiple GPUs, the model will be loaded automatically on GPUs and split the VRAM usage. That allows you to run Llama-2-7b (requires 14GB of GPU VRAM) on a setup like 2 GPUs (11GB VRAM each).

### Run on Low Memory GPU with 8 bit

If you do not have enough memory,  you can set up your `LOAD_IN_8BIT` as `True` in `.env`. This can reduce memory usage by around half with slightly degraded model quality. It is compatible with the CPU, GPU, and Metal backend.

Llama-2-7b with 8-bit compression can run on a single GPU with 8 GB of VRAM, like an Nvidia RTX 2080Ti, RTX 4080, T4, V100 (16GB).

### Run on Low Memory GPU with 4 bit

If you want to run 4 bit  Llama-2 model like `Llama-2-7b-Chat-GPTQ`,  you can set up your `LOAD_IN_4BIT` as `True` in `.env` like example `.env.7b_gptq_example`. 

Make sure you have downloaded the 4-bit model from `Llama-2-7b-Chat-GPTQ` and set the `MODEL_PATH` and arguments in `.env` file.

`Llama-2-7b-Chat-GPTQ` can run on a single GPU with 6 GB of VRAM.

## Credits

- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat
- https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ