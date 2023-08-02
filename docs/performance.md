# Benchmark Performance

## Performance on Nvidia GPU

| Model                             | Precision | Device | GPU VRAM | Speed (tokens/sec) | load time (s) |
| --------------------------------- | --------- | ---------- | ---------------------- | ---------------- | ---------------- |
| Llama-2-7b-chat-hf | 16 bit |  |  |              |              |
| Llama-2-7b-chat-hf          |   8bit   | NVIDIA RTX 2080 Ti    | 7.7 GB VRAM | 3.76 | 783.87 |
| Llama-2-7b-Chat-GPTQ        |   4bit   | NVIDIA RTX 2080 Ti    | 5.8 GB VRAM | 12.08 | 192.91 |
| Llama-2-7b-Chat-GPTQ        |   4bit   | NVIDIA GTX 1660 Super | 4.8 GB VRAM | 8.5   | 262.74        |
| Llama-2-13b-chat-hf               |   16 bit   |  |                  |                  |                  |
|  |  | |  | | |

## Performance on CPU / OpenBLAS / cuBLAS / CLBlast / Metal

| Model                             | Precision | Device | RAM / GPU VRAM | Speed (tokens/sec) | load time (s) |
| --------------------------------- | --------- | ---------- | ---------------------- | ---------------- | ---------------- |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit     | Intel i7-8700 | 4.5 GB RAM     | 5.70               | 71.48         |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit | Apple M2 CPU | 4.5 GB RAM | 10.49 | 0.14 |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit | Apple M2 Metal | 4.5 GB RAM | 10.51 | 0.64 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Intel i7-8700 | 5.4GB RAM     | 4.16               | 105.75 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Intel i7-9700 | 4.8GB RAM   | 4.2                 | 87.9        |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Apple M2 CPU | 5.4GB RAM | 5.28 | 0.20 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | Apple M2 Metal | 5.4GB RAM | 6.08 | 1.88 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | AMD Ryzen 9 5900HS | 4.1GB RAM | 6.01 | 0.15 |
| llama-2-7b-chat.ggmlv3.q8_0 | 8 bit | Intel i7-8700 | 8.6 GB RAM | 2.63 | 336.57 |
| llama-2-7b-chat.ggmlv3.q8_0 | 8 bit     | Intel i7-9700 | 7.6GB RAM   | 2.05              | 302.9    |


