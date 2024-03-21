---
license: mit
---

# scottsus/mamba-2.8b-custom

A Mamba model instruction-tuned using the [yamha/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset, then knowledge-injected with the [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs - Ovadia et al](https://arxiv.org/pdf/2312.05934.pdf) paper.

## Quick Start

1. Use a GPU with >= 60GB RAM and 60GB disk size
2. Verify machine is of the following type:
    - `$ cat /etc/os-release`: Ubuntu 22.04.3
    - `$ uname -m`: x86_64
3. Check for nvcc using `nvcc -V`, if exists skip this step
    - Install nvcc
        ```
        $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        $ sudo dpkg -i cuda-keyring_1.1-1_all.deb
        $ sudo apt-get update
        $ sudo apt-get -y install cuda-toolkit-12-4
        ```
    - Set to PATH
        ```
        $ export PATH=/usr/local/cuda/bin:$PATH
        $ export CUDA_HOME=/usr/local/cuda
        ```
    - Verify nvcc existence
        ```
        nvcc -V
        ```
4. Install requirements
    ```
    pip install -r requirements.txt
    ```
5. Run training
    ```
    python main.py
    ```
