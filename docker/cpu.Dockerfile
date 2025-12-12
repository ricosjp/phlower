# 3.11, 3.12
ARG USE_PYTHON_VERSION=3.12

# ----- 1. uv builder stage
FROM ubuntu:20.04 AS uv_builder
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh


# -----2. final stage
FROM python:${USE_PYTHON_VERSION}-slim
ENV DEBIAN_FRONTEND=noninteractive
ARG USE_PYTHON_VERSION
WORKDIR /workspace

RUN apt-get update \
 && apt-get install -y --no-install-recommends make git \
 && rm -rf /var/lib/apt/lists/*

 # Enable to use uv
COPY --from=uv_builder /root/.local/bin/uv /root/.local/bin/uvx /bin/
