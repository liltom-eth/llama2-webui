# Benchmark Performance

## Performance on Nvidia GPU

| Model                             | Precision | Device | GPU VRAM | Speed (tokens/sec) | load time (s) |
| --------------------------------- | --------- | ---------- | ---------------------- | ---------------- | ---------------- |
| Llama-2-7b-chat-hf | 16 bit |  |  |              |              |
| Llama-2-7b-chat-hf          |   8bit   | NVIDIA RTX 2080 Ti    | 7.7 GB VRAM | 3.76 | 641.36 |
| Llama-2-7b-Chat-GPTQ        |   4bit   | NVIDIA RTX 2080 Ti    | 5.8 GB VRAM | 18.85 | 192.91 |
| Llama-2-7b-Chat-GPTQ        |   4bit   | NVIDIA GTX 1660 Super | 4.8 GB VRAM | 8.5   | 262.74        |
| Llama-2-7b-Chat-GPTQ | 4 bit | Google Colab T4 | 5.8 GB VRAM | 18.19 | 37.44 |
| Llama-2-13b-chat-hf               |   16 bit   |  |                  |                  |                  |
|  |  | |  | | |

## Performance on CPU / OpenBLAS / cuBLAS / CLBlast / Metal

| Model                             | Precision | Device | RAM / GPU VRAM | Speed (tokens/sec) | load time (s) |
| --------------------------------- | --------- | ---------- | ---------------------- | ---------------- | ---------------- |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit     | Intel i7-8700 | 4.5 GB RAM     | 7.88               | 31.90         |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit | Apple M2 CPU | 4.5 GB RAM | 11.10 | 0.10 |
| llama-2-7b-chat.ggmlv3.q2_K | 2 bit | Apple M2 Metal | 4.5 GB RAM | 12.10 | 0.12 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Intel i7-8700 | 5.4 GB RAM     | 6.27            | 173.15 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Intel i7-9700 | 4.8 GB RAM   | 4.2                 | 87.9        |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | Apple M1 Pro CPU | 5.4 GB RAM | 17.90 | 0.18 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit     | Apple M2 CPU | 5.4 GB RAM | 13.70 | 0.13 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | Apple M2 Metal | 5.4 GB RAM | 12.60 | 0.10 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | AMD Ryzen 9 5900HS | 4.1 GB RAM | 6.01 | 0.15 |
| llama-2-7b-chat.ggmlv3.q4_0 | 4 bit | Intel vServer 4 threads, eth services | 8 GB RAM | 1.31 | 0.5|
| llama-2-7b-chat.ggmlv3.q8_0 | 8 bit | Intel i7-8700 | 8.6 GB RAM | 2.63 | 336.57 |
| llama-2-7b-chat.ggmlv3.q8_0 | 8 bit     | Intel i7-9700 | 7.6 GB RAM   | 2.05              | 302.9    |
|  |  |  |  |  |  |

