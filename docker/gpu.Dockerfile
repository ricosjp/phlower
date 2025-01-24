# 3.10, 3.11, 3.12
ARG USE_PYTHON_VERSION=3.11
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
      && apt install -y make \
      && pip install poetry

# Copy the project into the image
COPY dist/pyproject.toml /workspace/
COPY dist/Makefile /workspace/

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /workspace
RUN poetry config virtualenvs.in-project true && make install_cu124


FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
ARG USE_PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

COPY --from=builder /usr/bin/make /usr/bin/
COPY --from=builder /opt/conda/bin/poetry /opt/conda/bin
COPY --from=builder /opt/conda/lib/python${USE_PYTHON_VERSION}/site-packages /opt/conda/lib/python${USE_PYTHON_VERSION}/site-packages
COPY --from=builder /workspace /workspace
