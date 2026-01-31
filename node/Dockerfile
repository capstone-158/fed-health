FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

RUN pip install --no-cache-dir -r requirements.txt
RUN echo "all done but stuck"
ENTRYPOINT ["/app/start.sh"]

