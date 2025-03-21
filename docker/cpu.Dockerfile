# 3.10, 3.11, 3.12
ARG USE_PYTHON_VERSION

FROM python:${USE_PYTHON_VERSION}-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
ARG USE_PYTHON_VERSION

RUN apt update \
      && apt install -y make \
      && pip install poetry

# Copy the project into the image
COPY dist/pyproject.toml /workspace/
COPY dist/Makefile /workspace/

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /workspace
RUN poetry config virtualenvs.in-project true && make install


FROM python:${USE_PYTHON_VERSION}-slim
ENV DEBIAN_FRONTEND=noninteractive
ARG USE_PYTHON_VERSION
WORKDIR /workspace

COPY --from=builder /usr/bin/make /usr/bin/
COPY --from=builder /usr/local/bin/poetry /usr/local/bin
COPY --from=builder /usr/local/lib/python${USE_PYTHON_VERSION}/site-packages /usr/local/lib/python${USE_PYTHON_VERSION}/site-packages
COPY --from=builder /workspace /workspace

RUN chmod +x ./.venv/bin/activate
