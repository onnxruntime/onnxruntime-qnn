FROM docker-registry.qualcomm.com/library/ubuntu:22.04

ARG BUILD_USER
ARG BUILD_UID
ARG BUILD_GID

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    ca-certificates \
    libc++1 \
    libc++abi1 \
    patchelf \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
       python3.10 python3.10-dev python3.10-venv \
       python3.11 python3.11-dev python3.11-venv \
       python3.12 python3.12-dev python3.12-venv \
       python3.13 python3.13-dev python3.13-venv \
       python3.14 python3.14-dev python3.14-venv \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g $BUILD_GID $BUILD_USER && useradd -m -u $BUILD_UID -g $BUILD_USER $BUILD_USER

ARG ORT_NIGHTLY_BUILD=0
ENV ORT_NIGHTLY_BUILD=$ORT_NIGHTLY_BUILD

ARG ORT_VERSION_SUFFIX=
ENV ORT_VERSION_SUFFIX=$ORT_VERSION_SUFFIX

ENV AUDITWHEEL_PLAT=manylinux_2_35_x86_64
