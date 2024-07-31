# hadolint ignore=DL3008
# hadolint ignore=DL3009
ARG VARIANT="3.10.14"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} AS builder

WORKDIR /opt

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
ENV RYE_VERSION="0.37.0"

# hadolint ignore=DL3008,DL3009
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl zsh

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]
RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true

COPY ./.python-version ./pyproject.toml ./requirements* ./
RUN rye pin "$(cat .python-version)" && \
    rye sync --all-features



# FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

USER vscode

# RUN curl -sSf https://rye.astral.sh/get | RYE_VERSION="0.37.0" RYE_INSTALL_OPTION="--yes" bash
# ENV PATH=/home/vscode/.rye/shims:$PATH

# RUN echo "[[ -d .venv ]] && source .venv/bin/activate" >> /home/vscode/.bashrc
