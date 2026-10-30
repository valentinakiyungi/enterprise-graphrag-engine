FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt constraints.txt ./
# The cache mount keeps downloaded wheels between builds, so after a network failure a
# retry resumes instead of starting over. Long timeout + retries help on slow connections.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries=10 -r requirements.txt

COPY app ./app
COPY benchmark ./benchmark

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# PORT is honoured for platforms (Render, Railway, Fly) that inject it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
