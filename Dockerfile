FROM python:3.9-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

ARG UID
ARG GID
RUN if ! getent group $GID > /dev/null; then groupadd -g $GID user; fi && useradd -u $UID -g $GID -m user

RUN mkdir /app && chown -R $UID:$GID /app

USER user

WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
RUN pip install -e .

ENV http_proxy=""
ENV https_proxy=""