FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV REFRESH_INTERVAL_SECONDS=60
ENV PORT=5001

EXPOSE 5001

CMD gunicorn --workers 1 --threads 4 --bind 0.0.0.0:${PORT} --timeout 120 app:app
