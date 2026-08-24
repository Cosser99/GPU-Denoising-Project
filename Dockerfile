# CUDA Development Image
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Aggiorna il sistema e installa gli strumenti di sviluppo
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    gdb \
    pkg-config \
    wget \
    curl \
    unzip \
    vim \
    libopencv-dev \
    libtbb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Installazione Nsight
RUN apt-get install -y --no-install-recommends \
    gnupg \
    ca-certificates && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/$(dpkg --print-architecture) /" \
    > /etc/apt/sources.list.d/nvidia-devtools.list && apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && apt-get update && apt-get install -y nsight-systems-cli

# Cartella di lavoro
WORKDIR /workspace

# Compilatore di default
ENV CC=gcc
ENV CXX=g++

CMD ["/bin/bash"]