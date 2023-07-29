# Benchmark Performance

## Performance on Nvidia GPU

| Model                             | Precision | Device | GPU VRAM | Speed (tokens / sec) | load time (s) |
| --------------------------------- | --------- | ---------- | ---------------------- | ---------------- | ---------------- |
| Llama-2-7b-chat-hf | 16 bit |  |  |              |              |
| Llama-2-7b-chat-hf          |   8bit   | NVIDIA RTX 2080 Ti | 5.8 GB VRAM       | 3.76 | 783.87 |
| Llama-2-7b-Chat-GPTQ        |   4 bit   | NVIDIA RTX 2080 Ti | 7.7 GB VRAM        | 12.08 | 192.91 |
| Llama-2-13b-chat-hf               |   16 bit   |  |                  |                  |                  |
|  |  | |  | | |

## Performance on CPU / OpenBLAS / cuBLAS / CLBlast / Metal

| Model                             | Precision | Device | RAM / GPU VRAM | Speed (tokens / sec) | load time (s) |
| --------------------------------- | --------- | ---------- | ---------------------- | ---------------- | ---------------- |
| Llama-2-7B-Chat-GGML |   4 bit   | Intel i7-8700 | 5.1GB RAM       | 4.16 | 105.75 |
| Llama-2-7B-Chat-GGML |   4 bit   | Apple M1 CPU  |                |                  |                  |

