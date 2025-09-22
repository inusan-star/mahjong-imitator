FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

ARG UID
ARG GID
RUN groupadd -g $GID user && useradd -u $UID -g $GID -m user

RUN mkdir /app && chown -R user:user /app

USER user

WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
RUN pip install -e .