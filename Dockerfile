FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .
COPY config/constraints-main.txt config/constraints-main.txt
RUN python -m pip install --no-cache-dir pip==26.2.2 \
    && python -m pip install --no-cache-dir --group runtime -c config/constraints-main.txt

COPY app ./app
COPY docs ./docs
RUN mkdir -p /app/models /app/docs/data

ENTRYPOINT ["python", "-m", "app.pipeline"]
CMD ["--help"]
