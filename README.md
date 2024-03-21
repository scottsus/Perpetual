---
license: mit
---

# scottsus/mamba-2.8b-custom

A Mamba model instruction-tuned using the [yamha/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset, then knowledge-injected with the [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs - Ovadia et al](https://arxiv.org/pdf/2312.05934.pdf) paper.

## Quick Start

1. Use a GPU with >= 60GB RAM and 60GB disk size
2. Verify machine is of the following type:
    - Check distro
        ```
        cat /etc/os-release # Ubuntu 22.04.3
        ```
    - Check architecture
        ```
        uname -m            # x86_64
        ```
    - Check NVIDIA GPU
        ```
        nvidia-smi          # table of GPUs
        ```
3. Check for nvcc using `nvcc -V`, if exists skip this step
    - Install nvcc
        ```
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get -y install cuda-toolkit-12-4
        ```
    - Set to PATH
        ```
        export PATH=/usr/local/cuda/bin:$PATH
        export CUDA_HOME=/usr/local/cuda
        ```
    - Verify nvcc existence
        ```
        nvcc -V
        ```
4. Clone repository
    ```
    git clone git@hf.co:scottsus/mamba-2.8b-custom
    ```
5. Install requirements
    ```
    pip install -r requirements.txt
    ```
6. Run training
    ```
    python main.py
    ```

## Save Model Weights

Prerequisite: Huggingface account.

Since we're working with very large files ~5GB for each model weight, we need `git LFS` to help us.

1. Install git LFS
    ```
    sudo apt install git-lfs
    git lfs install
    ```
2. Enable large files
    ```
    huggingface-cli lfs-enable-largefiles .
    ```
3. Huggingface Credentials
    - Same way to authenticate yourself in GitHub.
4. Usual add, commit, push
    ```
    git add . && git commit -m "blah"
    git push
    ```
