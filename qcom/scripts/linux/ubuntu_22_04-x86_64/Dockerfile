FROM docker-registry.qualcomm.com/library/ubuntu:22.04

ARG BUILD_USER
ARG BUILD_UID
ARG BUILD_GID
ARG DEBIAN_FRONTEND=noninteractive

RUN groupadd -g $BUILD_GID $BUILD_USER && useradd -u $BUILD_UID -g $BUILD_USER $BUILD_USER

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential curl git

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.10 python3-distutils libpython3.10 python3.10-venv
RUN apt-get install -y python3.11 python3-distutils libpython3.11 python3.11-venv
RUN apt-get install -y python3.12 python3-distutils libpython3.12 python3.12-venv
RUN apt-get install -y python3.13 python3-distutils libpython3.13 python3.13-venv
RUN apt-get install -y python3.14 python3-distutils libpython3.14 python3.14-venv
