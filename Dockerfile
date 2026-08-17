FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docs ./docs
RUN mkdir -p /app/models /app/docs/data

ENTRYPOINT ["python", "-m", "app.pipeline"]
CMD ["--help"]
