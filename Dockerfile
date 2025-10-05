FROM python:3.9-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

ARG UID
ARG GID
RUN if ! getent group $GID > /dev/null; then groupadd -g $GID user; fi && useradd -u $UID -g $GID -m user

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["bash"]

USER user