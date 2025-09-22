FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN useradd --create-home user

RUN mkdir /app && chown -R user:user /app

USER user

WORKDIR /app

COPY --chown=user:user requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

RUN pip install -e .