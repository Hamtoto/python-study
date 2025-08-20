FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 설치 (Dockerfile에서와 동일)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8 \
                       cuda-cudart-12-8 \
                       cuda-npp-12-8 \
                       cuda-cufft-12-8 \
                       cuda-cublas-12-8 \
                       libcublas-dev-12-8 \
                       cuda-libraries-dev-12-8 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.0-1_all.deb

# OpenCV DEB 패키지 설치 테스트
COPY opencv-cuda-custom_4.13.0-1_amd64.deb /tmp/
RUN dpkg -i /tmp/opencv-cuda-custom_4.13.0-1_amd64.deb && \
    apt-get update && \
    apt-get install -f -y && \
    rm /tmp/opencv-cuda-custom_4.13.0-1_amd64.deb

# 설치 확인
RUN ldconfig && \
    pkg-config --modversion opencv4 || echo "OpenCV not found in pkg-config" && \
    ls -la /usr/local/lib/libopencv_cuda* || echo "CUDA libs not found"