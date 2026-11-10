FROM docker-registry.qualcomm.com/library/ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

ARG BUILD_USER
ARG BUILD_UID
ARG BUILD_GID

RUN apt-get update && apt-get install -y \
    build-essential ninja-build git wget curl patchelf \
    python3 python3-dev python3-venv python3-pip \
    lsb-release software-properties-common gnupg ca-certificates \
    lld-20 clang-20 llvm-20 \
    && rm -rf /var/lib/apt/lists/*

ENV ORT_LLVM_ROOT=/usr/lib/llvm-20/bin
ENV ORT_BUILD_PYTHON=python3

RUN groupadd -g $BUILD_GID $BUILD_USER && useradd -m -u $BUILD_UID -g $BUILD_USER $BUILD_USER

ARG ORT_NIGHTLY_BUILD=0
ENV ORT_NIGHTLY_BUILD=$ORT_NIGHTLY_BUILD

ARG ORT_VERSION_SUFFIX=
ENV ORT_VERSION_SUFFIX=$ORT_VERSION_SUFFIX
