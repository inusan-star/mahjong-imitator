FROM python:3.9-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN git clone -b mahjong-imitator --single-branch https://github.com/inusan-star/mjx-convert.git /mjx-convert
WORKDIR /mjx-convert
RUN make install && pip install .

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .

ENTRYPOINT ["bash"]