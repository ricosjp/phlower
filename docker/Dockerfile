FROM python:3.10-slim AS container
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
      && apt-get install -y make \
      && pip install poetry

