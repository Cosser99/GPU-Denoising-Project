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

# Cartella di lavoro
WORKDIR /workspace

# Compilatore di default
ENV CC=gcc
ENV CXX=g++

CMD ["/bin/bash"]