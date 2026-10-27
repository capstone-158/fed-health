FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends iputils-ping curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=100000 -r requirements.txt

ENV PYTHONPATH="/app"
CMD ["python3", "server.py"]